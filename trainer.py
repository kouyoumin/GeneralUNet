import torch
from torch import nn
import torchvision
import torch.nn.functional as F

import sys
sys.path.append('over9000')
from over9000.rangerlars import RangerLars
from over9000.adabelief import AdaBelief

from TorchModelModifier.modify_model_in import modify_first_conv_in_channels

import numpy as np
import time
import datetime
import os
import cv2

#from myresnet import resnext50_32x4d_fe
from myunet import UnetWithBackbone
from transforms import RandomResizedCrop2D, Resize2D, InPainting, OutPainting, Painting, LocalPixelShuffling, RandomWindow, CompressOutOfWindow, RandomGamma, RandomHorizontalFlip, Normalize, Compose
from autoencodedataset import DicomDataset, PatchDataset2D, AutoEncodeDataset


def main(args):
    outdir = time.strftime('%Y%m%d%H%M')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    os.system('cp *.py %s' % (outdir))
    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    # our dataset has two classes only - background and person
    num_classes = 1
    # use our dataset and defined transformations
    dcmdataset = DicomDataset('../data/202308mix', recursive=True, ext='.dcm')
    train_size = len(dcmdataset) - int(0.05*len(dcmdataset))
    valid_size = len(dcmdataset) - train_size
    train_dcmdataset, valid_dcmdataset = torch.utils.data.random_split(dcmdataset, [train_size, valid_size])
    #valid_pdataset = PatchDataset2D(valid_dcmdataset, 256/1120, 256/896, 0.5)
    
    train_general_transform = Compose([RandomResizedCrop2D(256, scale=(0.5, 1)), RandomHorizontalFlip()])
    train_input_transform = Compose([LocalPixelShuffling(), Painting(fill_mode='noise'), RandomWindow(), CompressOutOfWindow(), RandomGamma(), Normalize(dcmdataset.mean, dcmdataset.std)])
    #train_input_transform = Compose([RandomWindow(), CompressOutOfWindow(), RandomGamma(), Normalize(dcmdataset.mean, dcmdataset.std)])
    train_target_transform = Compose([CompressOutOfWindow()])

    rs = np.random.RandomState(0)
    valid_general_transform = Compose([Resize2D((288, 288), antialias=True)])
    valid_input_transform = Compose([LocalPixelShuffling(random_state=rs), Painting(random_state=rs, fill_mode='noise'), RandomWindow(random_state=rs), CompressOutOfWindow(), RandomGamma(random_state=rs), Normalize(dcmdataset.mean, dcmdataset.std)])
    #valid_input_transform = Compose([RandomWindow(random_state=rs), CompressOutOfWindow(), RandomGamma(random_state=rs), Normalize(dcmdataset.mean, dcmdataset.std)])
    valid_target_transform = Compose([CompressOutOfWindow()])

    train_dataset = AutoEncodeDataset(train_dcmdataset, train_general_transform, train_input_transform, train_target_transform)
    valid_dataset = AutoEncodeDataset(valid_dcmdataset, valid_general_transform, valid_input_transform, valid_target_transform)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    #backbone = resnext50_32x4d_fe(pretrained=True, grayscale=True)
    #model = UnetWithBackbone(backbone, {'layer4':'layer4', 'layer3':'layer3', 'layer2':'layer2', 'layer1':'layer1', 'relu':'relu'}, num_classes, scaler='deconv', res=True, droprate=0.5, shortcut_droprate=0.75, drop_func=F.dropout2d if args.dropout2d else F.dropout)
    arch = 'densenet121'
    backbone = torchvision.models.__dict__[arch](weights=torchvision.models.get_model_weights(arch).DEFAULT if args.pretrained else None)#pretrained=args.pretrained)
    modify_first_conv_in_channels(backbone, new_in_channels=1)
    return_layers = {'norm5':'norm5', 'transition3':'transition3', 'transition2':'transition2', 'transition1':'transition1', 'relu0':'relu0'}
    model = UnetWithBackbone(backbone.features, return_layers, num_classes, scaler='upsample' if args.upsample else 'deconv', res=args.residual, droprate=0., shortcut_droprate=0.5, drop_func=F.dropout2d, add_activation=nn.ReLU(), sigmoid=True)
    print(model)
    
    model.to(device)

    model_without_dp = model

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    #optimizer = AdaBelief(params, lr=args.lr, eps=1e-8, weight_decay=args.weight_decay, weight_decouple=False, rectify=False)
    #optimizer = RangerLars(params, lr=args.lr, weight_decay=args.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_gamma, patience=10, verbose=True)

    best_valid_metric = 1000
    best_train_metric = 1000
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_dp.load_state_dict(checkpoint['state_dict'])
        best_valid_metric = checkpoint['best_valid_metric']
        best_train_metric = checkpoint['best_train_metric']
        #optimizer.load_state_dict(checkpoint['optimizer'])
        #lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        #args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        evaluate(model, valid_data_loader, device=device)
        return

    model = nn.DataParallel(model)
    criterion_l1 = nn.L1Loss()
    criterion_mse = nn.MSELoss()
    criterion_train = [ {'name':'L1', 'lossfunction': criterion_l1, 'weight': 0.7},
                        {'name':'MSE', 'lossfunction': criterion_mse, 'weight': 0.3}]
    criterion_valid = [ {'name':'MSE', 'lossfunction': criterion_mse, 'weight': 1.0}]
    
    print("Start training")
    start_time = time.time()
    
    no_improve = 0
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        train_loss = train_epoch(model, optimizer, train_data_loader, criterion_train, device, epoch, args.print_freq, dcmdataset.mean, dcmdataset.std, outdir)
        epoch_train_end_time = time.time()

        valid_input_transform.transforms[0].random_state.seed(0)
        valid_loss = evaluate(model, valid_data_loader, criterion_valid, device, epoch, args.print_freq, dcmdataset.mean, dcmdataset.std, outdir)
        epoch_valid_end_time = time.time()
        print('Training time: %fs, Validation time: %fs' % (int(epoch_train_end_time - epoch_start_time), int(epoch_valid_end_time - epoch_train_end_time)))
        metric = valid_loss
        if train_loss < best_train_metric:
            torch.save({'state_dict': model_without_dp.state_dict(),
                        'best_valid_metric': metric,
                        'best_train_metric': train_loss,
                        'mean': dcmdataset.mean,
                        'std': dcmdataset.std}, os.path.join(outdir, 'best_model_train.pth'))
            #utils.save_on_master()
            print('Training loss decreased from', best_train_metric, 'to', train_loss, '- saving to best_model_train.pth')
            best_train_metric = train_loss
        if metric < best_valid_metric:
            torch.save({'state_dict': model_without_dp.state_dict(),
                        'best_valid_metric': metric,
                        'best_train_metric': train_loss,
                        'mean': dcmdataset.mean,
                        'std': dcmdataset.std}, os.path.join(outdir, 'best_model_valid.pth'))
            #utils.save_on_master()
            print('Metric decreased from', best_valid_metric, 'to', metric, '- saving to best_model_valid.pth')
            best_valid_metric = metric
            no_improve = 0
        else:
            no_improve += 1
            print('Metric =', metric, 'not decreased from', best_valid_metric, 'for', no_improve, 'epochs')
            if no_improve >= args.early_stop_epochs:
                print('Metric stopped improving for %d epochs, early stop training.' % (args.early_stop_epochs))
                break
        
        # update the learning rate
        lr_scheduler.step(metric)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    return


def train_epoch(model, optimizer, dataloader, criterions, device, epoch, print_freq, mean, std, outdir):
    losses = []
    model.train()
    for iteration, (input, target) in enumerate(dataloader):
        input = input.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = 0
        for c in criterions:
            loss += c['lossfunction'](output, target) * c['weight']
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if iteration % print_freq == 0:
            x = (((input[0][0].cpu().numpy() + mean) * std).clip(0,1) * 255).astype(np.uint8)
            t = (target[0][0].cpu().numpy().clip(0,1) * 255).astype(np.uint8)
            y = (output[0][0].cpu().detach().numpy().clip(0,1) * 255).astype(np.uint8)
            #print(output.max(), output.mean(), y.max())
            outfile = os.path.join(outdir, '%02dt_%03d.png' % (epoch, iteration))
            visualize(x, t, y, outfile)
            print('Epoch %02d, iter %d, Loss: %.4f, Average Loss: %.4f' % (epoch, iteration, loss.item(), np.average(losses)))
    
    return np.average(losses)


def evaluate(model, dataloader, criterions, device, epoch, print_freq, mean, std, outdir):
    losses = []
    print('Epoch %02d, start validation...' % (epoch))
    
    model.eval()
    with torch.no_grad():
        for iteration, (input, target) in enumerate(dataloader):
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            loss = 0
            for c in criterions:
                loss += c['lossfunction'](output, target) * c['weight']
            losses.append(loss.item())

            if iteration % print_freq == 0:
                x = (((input[0][0].cpu().numpy() + mean) * std).clip(0,1) * 255).astype(np.uint8)
                t = (target[0][0].cpu().numpy().clip(0,1) * 255).astype(np.uint8)
                y = (output[0][0].cpu().numpy().clip(0,1) * 255).astype(np.uint8)
                outfile = os.path.join(outdir, '%02dv_%03d.png' % (epoch, iteration))
                visualize(x, t, y, outfile)
                #print(output.max(), output.mean(), y.max())
                print('Epoch %02d, iter %d, Loss: %.4f, Average Loss: %.4f' % (epoch, iteration, loss.item(), np.average(losses)))
    
    return np.average(losses)


def visualize(input, target, output, outpath):
    img = np.concatenate((input, target, output), axis=1)
    cv2.imwrite(outpath, img)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--data-path', default='~/KVGH', help='dataset')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                        'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.5, type=float, help='decrease lr by a factor of lr-gamma')
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
    parser.add_argument(
        "--dropout2d",
        dest="dropout2d",
        help="dropout2d instead of dropout",
        action="store_true",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--upsample", action="store_true", help="Use upsample in decoder instead of deconvolution")
    parser.add_argument("--residual", action="store_true", help="Use residual shortcut instead of concatenation in unet")

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    main(args)