"""Contains definitions for VGG16 network. Adapted from https://github.com/arunmallya/packnet/blob/master/src/networks.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models

from functools import partial
from typing import Any, cast, Dict, List, Optional, Union

import torch
import torch.nn as nn



class View(nn.Module):
    """Changes view using a nn.Module."""

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view([-1,512])

 




#########################################################################################################################################################
#Modified Resnet
#########################################################################################################################################################
#!# Modified ResNet where all residual connections are 1x1 conv layers with batchnorm

# Source: https://github.com/huyvnphan/PyTorch_CIFAR10/blob/master/cifar10_models/resnet.py


# Source: https://github.com/huyvnphan/PyTorch_CIFAR10/blob/master/cifar10_models/resnet.py

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        affine=False,
    ):
        super(BasicBlock, self).__init__()
        print("BasicBlock affine: ", affine)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, affine=affine, track_running_stats=False)
        # self.bn1 = norm_layer(planes, track_running_stats=False)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, affine=affine, track_running_stats=False)
        # self.bn2 = norm_layer(planes, track_running_stats=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        identity = x
        out = self.conv1(x)
        out1 = self.bn1(out)
        out2 = self.relu(out1)

        out3 = self.conv2(out2)
        out4 = self.bn2(out3)

        if self.downsample is not None:
            identity = self.downsample(x)

        out5 = out4 + identity

        out6 = self.relu(out5)

        return out6









class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None,
        affine = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width, affine=affine, track_running_stats=False)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width, affine=affine, track_running_stats=False)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion, affine=affine, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out










class ModifiedResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        affine=False,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        print("Change updates")
        super(ModifiedResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.block = block

        self.affine=affine
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes, affine=self.affine, track_running_stats=False)
        # self.bn1 = norm_layer(self.inplanes, affine=False, track_running_stats=False)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = None
        # self.masks = {}


        for name, m in self.named_modules():
            # print("Module: ", name)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1, dilate: bool = False) :
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, self.affine
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    affine=self.affine,
                )
            )

        return nn.Sequential(*layers)





    ### The masks reflect which filters in the residuals need to be zeroed to avoid re-adding them to pruned filters.
    def forward(self, x, labels = False):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


    # ### The masks reflect which filters in the residuals need to be zeroed to avoid re-adding them to pruned filters.
    # def forward(self, x, labels = False):

    #     keys = [10,16,23,32,39,48,55,64]

    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)

    #     x = self.layer1[0](x)
    #     x = self.layer1[1](x)

    #     x = self.layer2[0](x)
    #     x = self.layer2[1](x)

    #     x = self.layer3[0](x)
    #     x = self.layer3[1](x)

    #     x = self.layer4[0](x)
    #     x = self.layer4[1](x)

    #     x = self.avgpool(x)
    #     x = torch.flatten(x, 1)
    #     x = self.classifier(x)

    #     return x


    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ResNet, self).train(mode)

        # Set the BNs to eval mode so that the running means and averages
        # do not update.
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()



def _modifiedresnet(arch, block, layers, useAffine, progress, device, **kwargs):
    print("Start")
    model = ModifiedResNet(block, layers, num_classes=10, affine=useAffine, **kwargs)
    return model


def modifiedresnet18(progress=True, affine=False, device="cpu", **kwargs):
    return _modifiedresnet("resnet18", BasicBlock, [2, 2, 2, 2], affine, progress, device, **kwargs)

def modifiedresnet50(progress=True, affine=False, device="cpu", **kwargs):
    return _modifiedresnet("resnet18", Bottleneck, [3, 4, 6, 3], affine, progress, device, **kwargs)


































#########################################################################################################################################################
# Modified VGG16
#########################################################################################################################################################


class View(nn.Module):
    """Changes view using a nn.Module."""

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)

class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 10, init_weights: bool = True, dropout: float = 0.0
    ):
        super().__init__()
        self.features = features
        self.classifier = None

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                # elif isinstance(m, nn.BatchNorm2d):
                    # print("No weights or bias to initialize in the batchnorm layers")
                    # nn.init.constant_(m.weight, 1)
                    # nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, labels = False):
        x = self.features(x)
        x = self.classifier(x)
        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, dropout=0.0):
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v, track_running_stats=False), nn.ReLU(inplace=True)]
                # layers += [conv2d, nn.BatchNorm2d(v, affine=False, track_running_stats=False), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    layers += [nn.AdaptiveAvgPool2d((1,1))]
    layers += [View(-1, 512)]
    layers += [nn.Linear(512, 512)] 
    layers += [nn.ReLU(True)]
    layers += [nn.Dropout(p=dropout)]
    #layers += [nn.Linear(4096, 4096)]
    #layers += [nn.ReLU(True)]
    #layers += [nn.Dropout(p=dropout)]

    return nn.Sequential(*layers)




 #          1   4         8    11       15    18   21        25  28    31        35   38   41   
    # "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def vgg16(cfg: str = "D", batch_norm: bool = True, dropout = 0.0, **kwargs: Any):
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), dropout=dropout, **kwargs)
    return model
    
    













