import copy
from multiprocessing.context import set_spawning_popen
from re import S
from turtle import forward
from typing_extensions import Required

import black
from numpy import block
from ..build import META_ARCH_REGISTRY
import torch

import torch
from torch import Tensor
import torch.nn as nn
from typing import Dict, Type, Any, Callable, Union, List, Optional
from torchvision import transforms
from torch.nn.modules.conv import _ConvNd
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
import torch.nn.functional as F


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

class Conv2d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode)
        
        self.avgpool = nn.AvgPool2d(kernel_size=3,stride=2,padding=1)
        self.dim1 = int(out_channels/3)
        self.dim2 = int(out_channels/3)
        self.dim3 = out_channels - self.dim1 -self.dim2
        self.w = nn.Parameter(torch.Tensor([0.1]),requires_grad=True)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor, originalImage) -> Tensor:
        if input.shape[2] == originalImage.shape[2]:
            if self.stride[0] == 1: 
                originalImage1 = originalImage[:,0,:,:].unsqueeze(dim=1).expand(originalImage.shape[0],self.dim1,originalImage.shape[2],originalImage.shape[3])
                originalImage2 = originalImage[:,1,:,:].unsqueeze(dim=1).expand(originalImage.shape[0],self.dim2,originalImage.shape[2],originalImage.shape[3])
                originalImage3 = originalImage[:,2,:,:].unsqueeze(dim=1).expand(originalImage.shape[0],self.dim3,originalImage.shape[2],originalImage.shape[3])
                originalImage = torch.cat([originalImage1,originalImage2,originalImage3],dim=1)
                conv_out = self._conv_forward(input, self.weight, self.bias)
                # con_mean = torch.max(conv_out)-torch.min(conv_out)
                # o_mean = torch.max(originalImage)-torch.min(originalImage)
                # w = (con_mean/(o_mean))
                #b = originalImage * (torch.mean(conv_out)/(3*torch.mean(originalImage)))
                return conv_out + originalImage * self.w
            else:
                originalImage = self.avgpool(originalImage)
                originalImage1 = originalImage[:,0,:,:].unsqueeze(dim=1).expand(originalImage.shape[0],self.dim1,originalImage.shape[2],originalImage.shape[3])
                originalImage2 = originalImage[:,1,:,:].unsqueeze(dim=1).expand(originalImage.shape[0],self.dim2,originalImage.shape[2],originalImage.shape[3])
                originalImage3 = originalImage[:,2,:,:].unsqueeze(dim=1).expand(originalImage.shape[0],self.dim3,originalImage.shape[2],originalImage.shape[3])
                originalImage = torch.cat([originalImage1,originalImage2,originalImage3],dim=1)
                conv_out = self._conv_forward(input, self.weight, self.bias)
                # con_mean = torch.max(conv_out)-torch.min(conv_out)
                # o_mean = torch.max(originalImage)-torch.min(originalImage)
                # w = (con_mean/(o_mean))
                #b = originalImage * (torch.mean(conv_out)/(3*torch.mean(originalImage)))
                return conv_out + originalImage * self.w
        else:
            while input.shape[2] < originalImage.shape[2]:
                originalImage = self.avgpool(originalImage)
            if self.stride[0] == 1: 
                originalImage1 = originalImage[:,0,:,:].unsqueeze(dim=1).expand(originalImage.shape[0],self.dim1,originalImage.shape[2],originalImage.shape[3])
                originalImage2 = originalImage[:,1,:,:].unsqueeze(dim=1).expand(originalImage.shape[0],self.dim2,originalImage.shape[2],originalImage.shape[3])
                originalImage3 = originalImage[:,2,:,:].unsqueeze(dim=1).expand(originalImage.shape[0],self.dim3,originalImage.shape[2],originalImage.shape[3])
                originalImage = torch.cat([originalImage1,originalImage2,originalImage3],dim=1)
                conv_out = self._conv_forward(input, self.weight, self.bias)
                # con_mean = torch.max(conv_out)-torch.min(conv_out)
                # o_mean = torch.max(originalImage)-torch.min(originalImage)
                # w = (con_mean/(o_mean))
                #b = originalImage * (torch.mean(conv_out)/(3*torch.mean(originalImage)))
                return conv_out + originalImage * self.w
            else:
                originalImage = self.avgpool(originalImage)
                originalImage1 = originalImage[:,0,:,:].unsqueeze(dim=1).expand(originalImage.shape[0],self.dim1,originalImage.shape[2],originalImage.shape[3])
                originalImage2 = originalImage[:,1,:,:].unsqueeze(dim=1).expand(originalImage.shape[0],self.dim2,originalImage.shape[2],originalImage.shape[3])
                originalImage3 = originalImage[:,2,:,:].unsqueeze(dim=1).expand(originalImage.shape[0],self.dim3,originalImage.shape[2],originalImage.shape[3])
                originalImage = torch.cat([originalImage1,originalImage2,originalImage3],dim=1)
                conv_out = self._conv_forward(input, self.weight, self.bias)
                # con_mean = torch.max(conv_out)-torch.min(conv_out)
                # o_mean = torch.max(originalImage)-torch.min(originalImage)
                # w = (con_mean/(o_mean))
                #b = originalImage * (torch.mean(conv_out)/(3*torch.mean(originalImage)))
                return conv_out + originalImage * self.w

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> Conv2d:
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=dilation, groups=groups, bias=False, dilation=dilation) 


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> Conv2d:
    """1x1 convolution"""
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        N=1
    ) -> None:
        super(BasicBlock, self).__init__()

        self.N = N
        if N==1:
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.downsample = downsample
            self.stride = stride
        else:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,stride=stride,padding=1,bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(planes, planes,kernel_size=3,stride=1,padding=1,bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.downsample = downsample
            self.stride = stride

    def forward(self, x: Tensor,originalImage) -> Tensor:
        if self.N == 1:
            identity = x
            out = self.conv1(x,originalImage)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out,originalImage)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x,originalImage)

            out += identity
            out = self.relu(out)
        else:
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample is not None:
                identity = self.downsample(x,originalImage)
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

class Downsample(nn.Module):
    def __init__(self,inplanes, planes, stride,N=1):
        super().__init__()
        self.N = N
        if N ==1:
            self.conv = conv1x1(inplanes, planes,stride)
            self.bn = nn.BatchNorm2d(planes)
        else:
            self.conv = nn.Conv2d(inplanes, planes,kernel_size=1,stride=stride)
            self.bn = nn.BatchNorm2d(planes)
    def forward(self,x,originalImage):
        if self.N == 1:
            x = self.conv(x,originalImage)
            x = self.bn(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
        return x

class make_layer(nn.Module):
    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]],inplanes, planes: int, blocks: int,
                    stride: int = 1, N=1):
        super().__init__()
        downsample = None
        if N == 1:
            if stride != 1 or inplanes != planes * block.expansion:
                downsample = Downsample(inplanes, planes * block.expansion, stride)
            self.block0 = block(inplanes, planes, stride, downsample)
            for i in range(1, blocks):
                self.add_module("block"+str(i),block(planes * block.expansion, planes))
        else:
            if stride != 1 or inplanes != planes * block.expansion:
                downsample = Downsample(inplanes, planes * block.expansion, stride, N)
            self.block0 = block(inplanes, planes, stride, downsample,N)
            for i in range(1, blocks):
                self.add_module("block"+str(i),block(planes * block.expansion, planes,N=N))

        self.blocks = blocks
        
    def forward(self,x,originalImage):   
        for i in range(0, self.blocks):
            x = self[i](x,originalImage)
        return x

    def __getitem__(self,index):
        
        assert index<4,"没有这么多网络层！"
        if index==0:
            if isinstance(self.block0,BasicBlock):
                return self.block0
        if index==1:
            return self.block1
        if index==2:
            return self.block2
        if index == 3:
            return self.block3



class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        inplanes = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = inplanes
        planes = copy.deepcopy(inplanes)
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
                        transforms.RandomCrop(32),
                        transforms.RandomHorizontalFlip(),
                        #transforms.Resize(224),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))

        self.transforms_evel = nn.Sequential(#transforms.Resize(224),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))


        self.conv1 = Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = make_layer(block,inplanes, planes, layers[0],N=1)
        inplanes = planes*block.expansion
        self.layer2 = make_layer(block,inplanes, 2*planes, layers[1], stride=2,N=1)
        inplanes = 2*planes*block.expansion
        self.layer3 = make_layer(block, inplanes, 4*planes, layers[2], stride=2,N=1)
        inplanes = 4*planes*block.expansion
        self.layer4 = make_layer(block, inplanes, 8*planes, layers[3], stride=2,N=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if layers[3] == 0:
            self.fc = nn.Linear(4*planes, num_classes)
        else:
            self.fc = nn.Linear(8*planes, num_classes)

        for m in self.modules():
            if isinstance(m, Conv2d):
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
        x = self.conv1(batch_images_tensor,batch_images_tensor)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x,batch_images_tensor) 
        x = self.layer2(x,batch_images_tensor) 
        x = self.layer3(x,batch_images_tensor) 
        #x = self.layer4(x,batch_images_tensor)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if self.training:
            #得到损失函数值
            batch_label = torch.tensor(batch_label,dtype=float).cuda()
            loss_fun = nn.CrossEntropyLoss()
            loss = loss_fun(x,batch_label.long())
            return loss,x
        else:
            #直接返回推理结果
            return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    inplance = 64,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers,inplanes=inplance, **kwargs)
    return model

def resnet20(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-20 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _resnet('resnet20', BasicBlock, [3, 3, 3, 0], pretrained,progress,inplance=16,**kwargs)
    model.conv1 = Conv2d(3, 16, kernel_size=3, stride=1, padding=1,bias=False)
    model.maxpool = nn.Identity()
    model.layer4 = nn.Identity()
    return model

def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

@META_ARCH_REGISTRY.register()
def Resnet20_addimage(cfg):
    return resnet20(num_classes=cfg.num_classes)

@META_ARCH_REGISTRY.register()
def Resnet18_addimage(cfg):
    return resnet18(num_classes=cfg.num_classes)

@META_ARCH_REGISTRY.register()
def Resnet34_add_image(cfg):
    return resnet34(num_classes=cfg.num_classes)

@META_ARCH_REGISTRY.register()
def Resnet50_addimage(cfg):
    return resnet50(num_classes=cfg.num_classes)

@META_ARCH_REGISTRY.register()
def Resnet101_addimage(cfg):
    return resnet101(num_classes=cfg.num_classes)

@META_ARCH_REGISTRY.register()
def Resnet152_addimage(cfg):
    return resnet152(num_classes=cfg.num_classes)

