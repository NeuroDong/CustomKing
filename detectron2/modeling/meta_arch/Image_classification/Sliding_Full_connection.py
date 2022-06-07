from re import S
from turtle import forward
import torch.nn as nn
import torch
from einops.layers.torch import Rearrange,Reduce
from ..build import META_ARCH_REGISTRY
import numpy as np
from typing import Dict, Type, Any, Callable, Union, List, Optional
from torch import Tensor
from torchvision import transforms

from torch.nn import init
import math

class MLPBlock(nn.Sequential):
    """Transformer MLP block."""

    def __init__(self, in_dim: int,out_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, (in_dim*out_dim)//(in_dim+out_dim))
        self.act = nn.GELU()
        #self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear((in_dim*out_dim)//(in_dim+out_dim), out_dim)
        #self.dropout_2 = nn.Dropout(dropout)

class Sliding_Full_connection(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,stride: int = 1,padding: int=0,bias: bool=True):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = padding
        self.bias = bias

        self.full_connect1 = nn.Linear(in_channel*kernel_size[0]*kernel_size[1],out_channel,bias=bias)
        #self.mlp = MLPBlock(in_channel*kernel_size[0]*kernel_size[1],out_channel)

        #张量变换
        self.tensorTransform1 = nn.Sequential(Rearrange('b n c h w -> b n (c h w)'))

        #self.conv = nn.Conv2d(in_channel,out_channel,kernel_size[0],stride=stride,padding=padding,bias=bias)

    def forward(self,x):
        H_out = (x.shape[2]+2*self.pad-self.kernel_size[0])//self.stride +1
        W_out = (x.shape[3]+2*self.pad-self.kernel_size[1])//self.stride +1

        x_pad = torch.zeros(x.shape[0],x.shape[1],x.shape[2]+2*self.pad,x.shape[3]+2*self.pad)
        x_pad[:,:,self.pad:self.pad+x.shape[2],self.pad:self.pad+x.shape[3]] = x
        vector_tensor = torch.zeros(x_pad.shape[0],H_out*W_out,x_pad.shape[1],self.kernel_size[0],self.kernel_size[1])
        h = 0
        for i in range(0,x_pad.shape[3]-self.kernel_size[0] + 1,self.stride):
            for j in range(0,x_pad.shape[2]-self.kernel_size[1] + 1,self.stride):
                vector_tensor[:,h] = x_pad[:,:,i:i+self.kernel_size[0],j:j+self.kernel_size[1]]
                h += 1
        vector_tensor = self.tensorTransform1(vector_tensor).cuda().clone().detach()
        vector_tmp = self.full_connect1(vector_tensor)
        #vector_tmp = self.mlp(vector_tensor)

        vector_tmp = torch.einsum('...ij->...ji',vector_tmp)
        assert H_out*W_out == h,"reshape不对"
        vector_tmp = vector_tmp.reshape(vector_tmp.shape[0],vector_tmp.shape[1],H_out,W_out)
        # vector_tmp = self.conv(x)
        return vector_tmp

def Fullcon3x3(in_planes: int, out_planes: int, stride: int = 1,padding: int=1) -> Sliding_Full_connection:
    """3x3 convolution with padding"""
    return Sliding_Full_connection(in_planes, out_planes, kernel_size=(3,3), stride=stride,padding=padding)

def Fullcon1x1(in_planes: int, out_planes: int, stride: int = 1) -> Sliding_Full_connection:
    """1x1 convolution"""
    return Sliding_Full_connection(in_planes, out_planes, kernel_size=(1,1), stride=stride, bias=False)

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
        self.conv1 = Fullcon3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Fullcon3x3(planes, planes)
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
        base_width: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = Fullcon1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = Fullcon3x3(width, width, stride)
        self.bn2 = norm_layer(width)
        self.conv3 = Fullcon1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
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

class Sliding_Full_connection_Network(nn.Module):

    def __init__(
        self,
        block: Type[Bottleneck],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Sliding_Full_connection_Network, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.base_width = width_per_group

        self.Fullcon1 = Sliding_Full_connection(3, self.inplanes, kernel_size=(7,7), stride=2, padding=3,bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.shotcut1 = nn.Sequential(Fullcon1x1(self.inplanes, 64 * block.expansion, 1),norm_layer(64 * block.expansion))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.shotcut2 = nn.Sequential(Fullcon1x1(self.inplanes, 128 * block.expansion, 2),norm_layer(128 * block.expansion))
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.shotcut3 = nn.Sequential(Fullcon1x1(self.inplanes, 256 * block.expansion, 2),norm_layer(256 * block.expansion))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.shotcut4 = nn.Sequential(Fullcon1x1(self.inplanes, 512 * block.expansion, 2),norm_layer(512 * block.expansion))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: Type[Bottleneck], planes: int, blocks: int,
                    stride: int = 1) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Fullcon1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,self.base_width, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,base_width=self.base_width,norm_layer=norm_layer))

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
        batch_images_tensor = torch.tensor(batch_images).float().cuda().clone().detach()

        # See note [TorchScript super()]
        x = self.Fullcon1(batch_images_tensor)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x) + self.shotcut1(x)
        x = self.layer2(x) + self.shotcut2(x)
        x = self.layer3(x) + self.shotcut3(x)
        x = self.layer4(x) + self.shotcut4(x)

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

def _Sliding_Full_connection_Network(
    arch: str,
    block: Type[Bottleneck],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> Sliding_Full_connection_Network:
    model = Sliding_Full_connection_Network(block, layers, **kwargs)
    return model

def Sliding_Fullcon18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> Sliding_Full_connection_Network:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _Sliding_Full_connection_Network('Sliding_Full_connection18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def Sliding_Fullcon50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> Sliding_Full_connection_Network:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _Sliding_Full_connection_Network('Sliding_Full_connection50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

@META_ARCH_REGISTRY.register()
def Sliding_Full_connection50(cfg):
    return Sliding_Fullcon50(num_classes=cfg.num_classes)


@META_ARCH_REGISTRY.register()
def Sliding_Full_connection18(cfg):
    return Sliding_Fullcon18(num_classes=cfg.num_classes)

        








