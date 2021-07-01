import torch
from torch import nn
import numpy as np
import time
import datetime
import os
import cv2

from myresnet import resnext50_32x4d_fe
from myunet import UnetWithBackbone
from transforms import RandomResizedCrop2D, Resize2D, InPainting, OutPainting, Painting, LocalPixelShuffling, RandomWindow, CompressOutOfWindow, RandomGamma, RandomHorizontalFlip, Normalize, Compose
from autoencodedataset import DicomDataset, PatchDataset2D, AutoEncodeDataset


def main(args):
    outdir = time.strftime('%Y%m%d%H%M')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    # our dataset has two classes only - background and person
    num_classes = 1
    # use our dataset and defined transformations
    dcmdataset = DicomDataset('~/hdd/KVGH', recursive=True, ext='')
    train_size = len(dcmdataset) - int(0.1*len(dcmdataset))
    valid_size = len(dcmdataset) - train_size
    train_dcmdataset, valid_dcmdataset = torch.utils.data.random_split(dcmdataset, [train_size, valid_size])
    valid_pdataset = PatchDataset2D(valid_dcmdataset, 256/1120, 256/896, 0.5)
    
    train_general_transform = Compose([RandomResizedCrop2D(256), RandomHorizontalFlip()])
    train_input_transform = Compose([LocalPixelShuffling(), Painting(), RandomWindow(), CompressOutOfWindow(), RandomGamma(), Normalize(dcmdataset.mean, dcmdataset.std)])
    train_target_transform = Compose([CompressOutOfWindow()])

    rs = np.random.RandomState(0)
    valid_general_transform = Compose([Resize2D((256, 256))])
    valid_input_transform = Compose([LocalPixelShuffling(random_state=rs), Painting(random_state=rs), RandomWindow(random_state=rs), CompressOutOfWindow(), RandomGamma(random_state=rs), Normalize(dcmdataset.mean, dcmdataset.std)])
    valid_target_transform = Compose([CompressOutOfWindow()])

    train_dataset = AutoEncodeDataset(train_dcmdataset, train_general_transform, train_input_transform, train_target_transform)
    valid_dataset = AutoEncodeDataset(valid_pdataset, valid_general_transform, valid_input_transform, valid_target_transform)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    backbone = resnext50_32x4d_fe(pretrained=True, grayscale=True)
    model = UnetWithBackbone(backbone, {'layer4':'layer4', 'layer3':'layer3', 'layer2':'layer2', 'layer1':'layer1', 'relu':'relu'}, num_classes, res=True)
    model.to(device)

    model_without_dp = model

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    #optimizer = AdaBelief(params, lr=args.lr, eps=1e-8, weight_decay=0.0001, weight_decouple=False, rectify=False)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_dp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        evaluate(model, valid_data_loader, device=device)
        return

    model = nn.DataParallel(model)
    criterion = nn.L1Loss()
    
    print("Start training")
    start_time = time.time()
    best_metric = 1000
    no_improve = 0
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        train_loss = train_epoch(model, optimizer, train_data_loader, criterion, device, epoch, args.print_freq)
        epoch_train_time = time.time() - epoch_start_time
        valid_input_transform.transforms[0].random_state.seed(0)
        valid_loss = evaluate(model, valid_data_loader, criterion, device, epoch, args.print_freq, dcmdataset.mean, dcmdataset.std, outdir)
        epoch_valid_time = time.time() - epoch_train_time

        print('Training time: %fs, Validation time: %fs' % (int(epoch_train_time), int(epoch_valid_time)))
        metric = valid_loss
        if metric < best_metric:
            torch.save({'state_dict': model_without_dp.state_dict(),
                        'metric': metric,
                        'mean': dcmdataset.mean,
                        'std': dcmdataset.std}, os.path.join(outdir, 'best_model.pth'))
            #utils.save_on_master()
            print('Metric decreased from', best_metric, 'to', metric, '- saving to best_model.pth.pth')
            best_metric = metric
            no_improve = 0
        else:
            no_improve += 1
            print('Metric =', metric, 'not decreased from', best_metric, 'for', no_improve, 'epochs')
            if no_improve >= args.early_stop_epochs:
                print('Metric stopped improving for %d epochs, early stop training.' % (args.early_stop_epochs))
                break
        
        # update the learning rate
        lr_scheduler.step(metric)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    return


def train_epoch(model, optimizer, dataloader, criterion, device, epoch, print_freq):
    losses = []
    model.train()
    for iteration, (input, target) in enumerate(dataloader):
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        loss = criterion(output, target)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if iteration % print_freq == 0:
            print('Epoch %02d, iter %d, Loss: %.4f, Average Loss: %.4f' % (epoch, iteration, loss.item(), np.average(losses)))
    
    return np.average(losses)


def evaluate(model, dataloader, criterion, device, epoch, print_freq, mean, std, outdir):
    losses = []
    print('Epoch %02d, start validation...' % (epoch))
    
    model.eval()
    with torch.no_grad():
        for e in range(10):
            for iteration, (input, target) in enumerate(dataloader):
                input = input.to(device)
                target = target.to(device)
                output = model(input)
                loss = criterion(output, target)
                losses.append(loss.item())

                if iteration % print_freq == 0:
                    x = (((input[0][0].cpu().numpy() + mean) * std).clip(0,1) * 255).astype(np.uint8)
                    t = (target[0][0].cpu().numpy().clip(0,1) * 255).astype(np.uint8)
                    y = (output[0][0].cpu().numpy().clip(0,1) * 255).astype(np.uint8)
                    img = np.concatenate((x, t, y), axis=1)
                    #print(img.shape)
                    cv2.imwrite('%s/%02d_%03d.png'%(outdir, epoch, iteration), img)
                    print('Epoch %02d, iter %d, Loss: %.4f, Average Loss: %.4f' % (epoch, iteration, loss.item(), np.average(losses)))
    
    return np.average(losses)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--data-path', default='/datasets01/COCO/022719/', help='dataset')
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--model', default='maskrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                        'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--rpn-score-thresh', default=None, type=float, help='rpn score threshold for faster-rcnn')
    parser.add_argument('--trainable-backbone-layers', default=5, type=int,
                        help='number of trainable layers of backbone')
    parser.add_argument('--early-stop-epochs', default=50, type=int)
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    parser.add_argument(
        "--maskrcnn",
        dest="maskrcnn",
        help="Instance Segmentation",
        action="store_true",
    )
    parser.add_argument("--local_rank", type=int, default=0)

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    main(args)