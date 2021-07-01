from json import decoder
import torch
from torch import Tensor
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, model_urls
from torchvision.models.utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Tuple, Optional


class ResNetFE(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        unet=False,
        **kwargs: Any):
        super(ResNetFE, self).__init__(block, layers, **kwargs)
    
    #def _load_state_dict(self, state_dict, strict=True):
    #    self.load_state_dict(state_dict, strict)

    def forward(self, x: Tensor, fe=None, interpolate=False) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        print('conv1', x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print('maxpool', x.shape)
        x = self.layer1(x)
        print('layer1', x.shape)
        x = self.layer2(x)
        print('layer2', x.shape)
        x = self.layer3(x)
        print('layer3', x.shape)
        x = self.layer4(x)
        print('layer4', x.shape)
        if isinstance(fe, bool):
            return x
        elif isinstance(fe, Tuple) and len(fe)==2:
            if interpolate:
                return nn.functional.interpolate(x, size=fe, mode='bilinear', align_corners=True)
            else:
                return nn.functional.adaptive_max_pool2d(x, fe)
        else:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x


def _resnetfe(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNetFE:
    model = ResNetFE(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnext50_32x4d_fe(pretrained: bool = True, progress: bool = True, grayscale: bool = False, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    model = _resnetfe('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)
    if grayscale:
        conv1_state_dict = model.conv1.state_dict()
        #print(conv1_state_dict['weight'].shape)
        conv1_state_dict['weight'] = conv1_state_dict['weight'].sum(dim=1, keepdim=True)
        model.conv1 = torch.nn.Conv2d(1, conv1_state_dict['weight'].shape[0], kernel_size=7, stride=2, padding=3, bias=False)
        model.conv1.load_state_dict(conv1_state_dict)
    
    return model


if __name__ == '__main__':
    model = resnext50_32x4d_fe(grayscale=True)
    print(model)

    t = torch.rand((1,1,128,128))
    print(t.shape)
    #print(model(t).shape)
    r1=model(t, fe=True)
    r2=model(t, fe=(1,2))
    r3=model(t, fe=(1,2), interpolate=True)
    print(r1.shape)
    print(r2.shape)
    print(r1, r2, r3)
    print(r2-r3)
