import torch
from torch import nn
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import BasicBlock
from typing import Tuple, List, Dict, Optional


class ExpansionBlock(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    def __init__(self, lowres_inplanes, shortcut_inplanes, outplanes, scale=2, scaler='upsample', groups=1,
                 base_width=64, norm_layer=None, res=False, droprate=0, shortcut_droprate=0.5, drop_func=F.dropout):
        super(ExpansionBlock, self).__init__()
        self.res = res
        self.droprate = droprate
        self.shortcut_droprate = shortcut_droprate
        self.drop_func = drop_func
        self.outplanes = outplanes
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        if scaler == 'deconv':
            self.scale0 = nn.ConvTranspose2d(lowres_inplanes, outplanes, kernel_size=scale, stride=scale, groups=outplanes, bias=False)
        else:
            self.scale0 = nn.Sequential(nn.Upsample(scale_factor=scale, mode='bilinear'), nn.Conv2d(lowres_inplanes, outplanes, kernel_size=1, bias=False))
        self.bn0 = norm_layer(outplanes)
        if shortcut_inplanes > 0:
            if res:
                self.conv1 = nn.Conv2d(shortcut_inplanes, outplanes, kernel_size=1, stride=1, groups=1, bias=False)
            else:
                self.conv1 = nn.Conv2d(shortcut_inplanes + outplanes, outplanes, kernel_size=1, stride=1, groups=1, bias=False)
                
            self.bn1 = norm_layer(outplanes)
        self.resblock = BasicBlock(outplanes, outplanes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=norm_layer)

    #@autocast
    def forward(self, lowres_in, shortcut_in=None):
        lowres_in = self.drop_func(lowres_in, p=self.droprate, training=self.training)
        out = self.bn0(self.scale0(lowres_in))
        if shortcut_in is not None:
            shortcut_in = self.drop_func(shortcut_in, p=self.shortcut_droprate, training=self.training)
            if out.shape[2:] != shortcut_in.shape[2:]:
                out = F.interpolate(out, shortcut_in.size()[2:], mode='bilinear')
            if self.res:
                out = self.bn1(self.conv1(shortcut_in)) + out
            else:
                out = torch.cat((out, shortcut_in), 1)
                out = self.bn1(self.conv1(out))
        out = self.resblock(out)

        return out


class OutputTransition(nn.Module):
    def __init__(self, inplanes, n_labels, sigmoid=True):
        super(OutputTransition, self).__init__()
        modules = [nn.Conv2d(inplanes, n_labels, kernel_size=1)]
        if sigmoid:
            modules.append(nn.Sigmoid())
        self.final_conv = nn.Sequential(*modules)

    #@autocast()
    def forward(self, x):
        out = self.final_conv(x)
        return out


class UnetDecoder(nn.Module):
    def __init__(
        self,
        in_channels_list: List[int],
        scale_list: List[int],
        out_channels: int,
        norm_layer = None,
        scaler='upsample',
        res = False,
        droprate = 0,
        shortcut_droprate = 0.5,
        drop_func = F.dropout,
        no_shortcut = False,
        multiscale_out = False,
        relu = True,
        sigmoid = True
    ):
        super(UnetDecoder, self).__init__()
        self.no_shortcut = no_shortcut
        self.multiscale_out = multiscale_out
        self.blocks = nn.ModuleList()
        self.outtrans = nn.ModuleList()
        if relu:
            self.relu = nn.ReLU()
        else:
            self.relu = nn.Identity()
        num_inputs = len(in_channels_list)
        for idx, in_channels in enumerate(in_channels_list):
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            if idx == 0:
                block_module = ExpansionBlock(in_channels, 0 if no_shortcut else in_channels_list[idx+1], in_channels_list[idx+1], scale=scale_list[idx], scaler=scaler, norm_layer=norm_layer, res=res, droprate=droprate, shortcut_droprate=shortcut_droprate, drop_func=drop_func)
            elif idx < num_inputs - 1:
                block_module = ExpansionBlock(in_channels, 0 if no_shortcut else in_channels_list[idx+1], in_channels_list[idx+1], scale=scale_list[idx], scaler=scaler, norm_layer=norm_layer, res=res, droprate=0, shortcut_droprate=shortcut_droprate, drop_func=drop_func)
            else:
                block_module = ExpansionBlock(in_channels, 0, in_channels, scale=scale_list[idx], scaler=scaler, norm_layer=norm_layer, res=res, droprate=0, shortcut_droprate=shortcut_droprate, drop_func=drop_func)
            
            self.blocks.append(block_module)
        if multiscale_out:
            for idx in range(len(self.blocks)):
                self.outtrans.append(OutputTransition(self.blocks[idx].outplanes, out_channels, sigmoid=sigmoid))
        else:
            self.outtrans.append(OutputTransition(self.blocks[-1].outplanes, out_channels, sigmoid=sigmoid))
        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            #print(m)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    
    #@autocast
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        num_inputs = len(x)
        num_layers = len(self.blocks)
        x = list(x.values())[::-1]
        '''out = x[0]
        for idx in range(num_inputs - 1):
            out = self.blocks[idx](out, None if self.no_shortcut else x[idx+1])
            #print(idx)
        #print(idx)
        while idx < num_layers - 1:
            idx += 1
            #print(idx)
            out = self.blocks[idx](out)'''
        outs = []
        for idx in range(num_layers):
            shortcut = x[idx+1] if idx < num_inputs - 1 and not self.no_shortcut else None
            #print(idx, shortcut.shape if shortcut is not None else 'None')
            out = self.blocks[idx](self.relu(x[0]) if idx == 0 else out, shortcut)
            if self.multiscale_out:
                outs.append(self.outtrans[idx](out))
            elif idx == num_layers - 1:
                return self.outtrans[0](out)
        
        return outs


class UnetWithBackbone(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.
    Args:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
    Attributes:
        out_channels (int): the number of channels in the FPN
    """
    def __init__(self, backbone, return_layers, out_channels, scaler='upsample', res=False, droprate=0.5, shortcut_droprate=0.5, no_shortcut=False, relu=True, sigmoid=True, classifier_out=False, multiscale_out=False, drop_func=F.dropout):
        super(UnetWithBackbone, self).__init__()

        self.backbone = backbone
        self.body = IntermediateLayerGetter(self.backbone, return_layers=return_layers)
        self.classifier_out = classifier_out
        self.multiscale_out = multiscale_out
        in_channels_list, scale_list = self._get_channel_scale_info()
        print('in_channels_list', in_channels_list)
        print('scale_list', scale_list)
        if classifier_out:
            self.classifier = nn.Sequential(nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(in_channels_list[0], out_channels))
        self.decoder = UnetDecoder(in_channels_list, scale_list, out_channels, scaler=scaler, res=res, droprate=droprate, shortcut_droprate=shortcut_droprate, no_shortcut=no_shortcut, multiscale_out=multiscale_out, relu=relu, sigmoid=sigmoid, drop_func=drop_func)
        '''self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )'''
        self.out_channels = out_channels

    def _get_channel_scale_info(self):
        ch_list = []
        sc_list = []
        
        training_state = self.training
        self.eval()
        
        with torch.no_grad():
            for p in self.body.parameters():
                dummy = torch.zeros((1,p.shape[1],128,128))
                print('dummy', dummy.shape)
                break
            feats = self.body(dummy)
            #for key in feats:
            #    print(feats[key].shape)
            feats = list(feats.values())[::-1]
            for idx in range(len(feats)):
                print(feats[idx].shape)
                ch_list.append(feats[idx].shape[1])
                if idx > 0:
                    sc_list.append(feats[idx].shape[2]//feats[idx-1].shape[2])
            sc_list.append(128//feats[-1].shape[2])
        
        if training_state:
            self.train()
        
        return ch_list, sc_list  
    
    #@autocast
    def forward(self, x):
        enc = self.body(x)
        dec = self.decoder(enc)
        if self.classifier_out:
            embedding = list(enc.values())[-1]
            return self.classifier(embedding), dec
        return dec


if __name__ == '__main__':
    #from torchvision.models.resnet import resnext50_32x4d
    #backbone = resnext50_32x4d(pretrained=True)
    #model = UnetWithBackbone(backbone, {'layer4':'layer4', 'layer3':'layer3', 'layer2':'layer2', 'layer1':'layer1', 'relu':'relu'}, 1, res=False, classifier_out=False, multiscale_out=True)

    from torchvision.models.densenet import densenet121
    backbone = densenet121(pretrained=True)
    model = UnetWithBackbone(backbone.features, {'denseblock4':'denseblock4', 'transition3':'transition3', 'transition2':'transition2', 'transition1':'transition1', 'relu0':'relu0'}, 1, res=False, classifier_out=True, multiscale_out=True)
    input = torch.rand((1,3,256,256))
    output = model(input)
    
    if isinstance(output, Tuple):
        cls_output, mask_output = output
        print('classifier_out:', cls_output.shape)
    else:
        mask_output = output
    
    if isinstance(mask_output, List):
        for idx, mask in enumerate(mask_output):
            print('mask_out:', idx, mask.shape)
    else:
        print('mask_out:', mask_output.shape)
    #for key in output:
    #    print(key, output[key].shape)