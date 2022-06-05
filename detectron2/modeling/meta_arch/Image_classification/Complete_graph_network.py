from multiprocessing.context import set_spawning_popen
from re import S
from ..build import META_ARCH_REGISTRY
import torch
import torch.nn as nn

import torch
from torch import Tensor
import torch.nn as nn
from typing import Dict, Type, Any, Callable, Union, List, Optional
from torchvision import transforms


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
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


class complete_graph_network(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(complete_graph_network, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 128
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.transforms_train = nn.Sequential(transforms.Pad(4),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))

        self.transforms_evel = nn.Sequential(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer12 = self._make_layer(block, 128, 128, 2,stride=2)
        self.layer13 = nn.Sequential(self._make_layer(block, 128, 128, 2,stride=2),
                        self._make_layer(block, 128, 128, 2,stride=2))
        self.layer14 = nn.Sequential(self._make_layer(block, 128, 128, 2,stride=2),
                        self._make_layer(block, 128, 128, 2,stride=2),
                        self._make_layer(block, 128, 128, 2,stride=2))
        self.layer15 = nn.Sequential(self._make_layer(block, 128, 128, 2,stride=2),
                        self._make_layer(block, 128, 128, 2,stride=2),
                        self._make_layer(block, 128, 128, 2,stride=2),
                        self._make_layer(block, 128, 128, 2,stride=2))
          
        self.layer23 = self._make_layer(block, 128, 128, 2, stride=2)
        self.layer24 = nn.Sequential(self._make_layer(block, 128, 128, 2, stride=2),
                        self._make_layer(block, 128, 128, 2, stride=2))
        self.layer25 = nn.Sequential(self._make_layer(block, 128, 128, 2, stride=2),
                        self._make_layer(block, 128, 128, 2, stride=2),
                        self._make_layer(block, 128, 128, 2, stride=2))

        self.layer34 = self._make_layer(block, 128, 128, 2, stride=2)
        self.layer35 = nn.Sequential(self._make_layer(block, 128, 128, 2, stride=2),
                        self._make_layer(block, 128, 128, 2, stride=2))
    
        self.layer45 = self._make_layer(block, 128, 128, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Sequential(nn.Linear(128,512),nn.LayerNorm(512),nn.ReLU())
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], inplanes: int,planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        inplanes2 = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes2, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, data: Dict) -> Tensor:
        
        #------------------预处理(data里面既含有image、label、width、height信息。)-----------------#
        batchsize = len(data)
        batch_images = []
        batch_label = []
        for i in range(0,batchsize,1):
            batch_images.append(data[i]["image"])
            batch_label.append(int(float(data[i]["y"])))
        batch_images=[image.tolist() for image in batch_images]
        batch_images_tensor = torch.tensor(batch_images,dtype=torch.float).cuda().clone().detach()

        if self.training:
            batch_images_tensor = self.transforms_train(batch_images_tensor)
        else:
            batch_images_tensor = self.transforms_evel(batch_images_tensor)

        # See note [TorchScript super()]
        x = self.conv1(batch_images_tensor)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x2 = self.layer12(x)
        x3 = self.layer13(x) + self.layer23(x2)
        x4 = self.layer14(x) + self.layer24(x2) + self.layer34(x3)
        x5 = self.layer15(x) + self.layer25(x2) + self.layer35(x3) + self.layer45(x4)

        x = self.avgpool(x5)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc(x)

        if self.training:
            #得到损失函数值
            batch_label = torch.tensor(batch_label,dtype=float).cuda()
            loss_fun = nn.CrossEntropyLoss()
            loss = loss_fun(x,batch_label.long())
            return loss
        else:
            #直接返回推理结果
            return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _complete_graph_network(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> complete_graph_network:
    model = complete_graph_network(block, layers, **kwargs)
    return model


def complete_graph_network18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> complete_graph_network:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _complete_graph_network('complete_graph_network18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


@META_ARCH_REGISTRY.register()
def Complete_graph_network18(cfg):
    return complete_graph_network18(num_classes=cfg.num_classes)


