from copy import deepcopy
from http.client import NETWORK_AUTHENTICATION_REQUIRED
from multiprocessing.context import set_spawning_popen
from pyexpat.errors import XML_ERROR_FEATURE_REQUIRES_XML_DTD
from re import S
from turtle import forward
from numpy import float32
import torch.nn as nn

from ..build import META_ARCH_REGISTRY
import torch.nn as nn
import torch.nn.init as init
import math

import torch
from torch import Tensor
from typing import Dict, Type, Any, Callable, Union, List, Optional
from torchvision import transforms
import time
from concurrent.futures import ThreadPoolExecutor,as_completed, process,wait,ALL_COMPLETED


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1)

def conv7x7(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    #return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3)
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1))

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride)

class BasicBlock2(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        inplanes = int(inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.conv1_1 = conv7x7(inplanes, int(planes/2))
        self.bn1 = norm_layer(int(planes/2))
        if int(planes) % 2 ==0:
            self.conv1_2 = conv3x3(inplanes, int(planes/2))
            self.bn2 = norm_layer(int(planes/2))
        else:
            self.conv1_2 = conv3x3(inplanes, int(planes/2)+1)
            self.bn2 = norm_layer(int(planes/2)+1)

        self.conv2_1 = conv7x7(inplanes, int(planes/2),stride)
        self.bn3 = norm_layer(int(planes/2))
        if int(planes) % 2 ==0:
            self.conv2_2 = conv3x3(inplanes, int(planes/2), stride)
            self.bn4 = norm_layer(int(planes/2))
        else:
            self.conv2_2 = conv3x3(inplanes, int(planes/2)+1, stride)
            self.bn4 = norm_layer(int(planes/2)+1)

    def forward(self, x: Tensor) -> Tensor:
        
        out1_1 = self.relu(self.bn1(self.conv1_1(x)))
        out1_2 = self.relu(self.bn2(self.conv1_2(x)))
        x = torch.cat([out1_1,out1_2],dim=1)

        out2_1 = self.relu(self.bn3(self.conv2_1(x)))
        out2_2 = self.relu(self.bn4(self.conv2_2(x)))
        x = torch.cat([out2_1,out2_2],dim=1)
        return x

    def reset_parameters(self):
        for m in self.conv1_1:
            if not isinstance(m,nn.ReLU):
                m.reset_parameters()
        self.bn1.reset_parameters()
        self.conv1_2.reset_parameters()
        self.bn2.reset_parameters()

        for m in self.conv2_1:
            if not isinstance(m,nn.ReLU):
                m.reset_parameters()
        self.bn3.reset_parameters()
        self.conv2_2.reset_parameters()
        self.bn4.reset_parameters()

class BasicBlock3(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock3, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        inplanes = int(inplanes)
        dim1 = int(inplanes/4)
        dim2 = int(inplanes/4)
        dim3 = int(inplanes/4)
        dim4 = inplanes - dim1 - dim2 - dim3

        self.conv1 = nn.Sequential(conv3x3(inplanes,dim1),nn.BatchNorm2d(dim1),nn.ReLU(dim1))
        self.conv2 = nn.Sequential(conv3x3(dim1,dim2),nn.BatchNorm2d(dim2),nn.ReLU(dim2))
        self.conv3 = nn.Sequential(conv3x3(dim1+dim2,dim3),nn.BatchNorm2d(dim3),nn.ReLU(dim3))
        self.conv4 = nn.Sequential(conv3x3(dim1+dim2+dim3,dim4),nn.BatchNorm2d(dim4),nn.ReLU(dim4))


    def forward(self, x: Tensor) -> Tensor:
        dim1_out = self.conv1(x)
        dim2_out = torch.cat([dim1_out,self.conv2(dim1_out)],dim=1)
        dim3_out = torch.cat([dim2_out,self.conv3(dim2_out)],dim=1)
        dim4_out = torch.cat([dim3_out,self.conv4(dim3_out)],dim=1)
        return dim4_out

    def reset_parameters(self):
        for m in self.conv1:
            if not isinstance(m,nn.ReLU):
                m.reset_parameters()
        for m in self.conv2:
            if not isinstance(m,nn.ReLU):
                m.reset_parameters()
        for m in self.conv3:
            if not isinstance(m,nn.ReLU):
                m.reset_parameters()
        for m in self.conv4:
            if not isinstance(m,nn.ReLU):
                m.reset_parameters()

def mp_fun(*args):
    network = args[0]
    x = args[1]
    return network(x).detach()

def judge(num):
    '''
    判断num是否是2的整数次幂
    '''
    result = num & (num-1)
    if result == 0:
        return True
    else:
        return False

class genetate_network(nn.Module):
    def __init__(self,block,dim,base_dim):
        super().__init__()
        assert judge(int(dim/base_dim)),"不满足dim=base_dim*2**(n-1)"
        self.dim = dim
        self.base_dim = base_dim

        if dim > base_dim + 2:

            self.is_base = False
            self.relu = nn.ReLU(inplace=True)

            self.convleft1 = conv1x1(int(dim),int(dim/2))
            self.bn_left1 = nn.BatchNorm2d(int(dim/2))
            self.network1_1 = genetate_network(block,dim/2,base_dim)
            if int(dim)%2==0:
                self.convright1 = conv3x3(int(dim),int(dim/2))
                self.bn_right1 = nn.BatchNorm2d(int(dim/2))
                self.network1_2 = genetate_network(block,dim/2,base_dim)
            else:
                self.convright1 = conv3x3(int(dim),int(dim/2)+1)
                self.bn_right1 = nn.BatchNorm2d(int(dim/2)+1)
                self.network1_2 = genetate_network(block,(dim/2)+1,base_dim)
            self.bn1 = nn.BatchNorm2d(int(dim))

            self.convleft2 = conv1x1(int(dim),int(dim/2))
            self.bn_left2 = nn.BatchNorm2d(int(dim/2))
            self.network2_1 = genetate_network(block,dim/2,base_dim)
            if int(dim)%2==0:
                self.convright2 = conv3x3(int(dim),int(dim/2))
                self.bn_right2 = nn.BatchNorm2d(int(dim/2))
                self.network2_2 = genetate_network(block,dim/2,base_dim)
            else:
                self.convright2 = conv3x3(int(dim),int(dim/2)+1)
                self.bn_right2 = nn.BatchNorm2d(int(dim/2)+1)
                self.network2_2 = genetate_network(block,(dim/2)+1,base_dim)
            self.bn2 = nn.BatchNorm2d(int(dim))

        else:
            self.is_base = True
            self.network_base = block(dim,dim,2)

    def forward(self,x):
        if self.is_base:
            x = self.network_base(x)
        else:
            x_left = self.network1_1(self.relu(self.bn_left1(self.convleft1(x))))
            x_right = self.network1_2(self.relu(self.bn_right1(self.convright1(x))))
            x = self.relu(self.bn1(torch.cat([x_left,x_right],dim=1)))
            x_left = self.network2_1(self.relu(self.bn_left2(self.convleft2(x))))
            x_right = self.network2_2(self.relu(self.bn_right2(self.convright2(x))))
            x = self.relu(self.bn2(torch.cat([x_left,x_right],dim=1)))
        return x

    def __setitem__(self, str, v):
        if "network_base" == str :
            if isinstance(self.network_base,BasicBlock2):
                self.network_base = v
        
    def __getitem__(self,str):
        if "network1" == str:
            return self.network1
        if "network2" == str:
            return self.network2
        if "network3" == str:
            return self.network3
        if "network4" == str:
            return self.network4
        if "network_base" == str and isinstance(self.network_base,genetate_network):
            return self.network_base

    def reset_parameters(self):
        if self.dim > self.base_dim + 2:
            self.convleft1.reset_parameters()
            self.bn_left1.reset_parameters()
            self.network1_1.reset_parameters()
            self.convright1.reset_parameters()
            self.bn_right1.reset_parameters()
            self.network1_2.reset_parameters()
            self.bn1.reset_parameters()

            self.convleft2.reset_parameters()
            self.bn_left2.reset_parameters()
            self.network2_1.reset_parameters()
            self.convright2.reset_parameters()
            self.bn_right2.reset_parameters()
            self.network2_2.reset_parameters()
            self.bn2.reset_parameters()
        else:
            self.network_base.reset_parameters()

@META_ARCH_REGISTRY.register()
class Fractal_network(nn.Module):
    def __init__(self,cfg,dim = 487,base_dim = 243):
        super().__init__()
        self.dim = dim
        self.base_dim = base_dim
        self.conv1 = nn.Conv2d(3, dim-3, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(dim-3)

        self.relu = nn.ReLU(inplace=True)        
        self.network = genetate_network(BasicBlock3,dim=dim,base_dim=base_dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dim , cfg.num_classes)

        self.transforms_train = nn.Sequential(transforms.Pad(4),
                        transforms.RandomCrop(32),
                        transforms.RandomHorizontalFlip(),
                        #transforms.Resize(224),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))

        self.transforms_evel = nn.Sequential(#transforms.Resize(224),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))

    def forward(self,data):
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
        
        # x_left = self.relu(self.bn1_1(self.conv1_1(batch_images_tensor)))
        # x_right = self.relu(self.bn1_2(self.conv1_2(batch_images_tensor)))
        # x = torch.cat([x_left,x_right],dim=1)

        x = self.relu(self.bn1(self.conv1(batch_images_tensor)))
        x = torch.cat([x,batch_images_tensor],dim=1)
        x = self.network(x)
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

    def __setitem__(self, str, v):
        if "network"==str:
            self.network = v

    def __getitem__(self,str):
        if "network" in str:
            return self.network

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.bn1.reset_parameters()
        # for m in self.conv1_2:
        #     if not isinstance(m,nn.ReLU):
        #         m.reset_parameters()
        # self.bn1_2.reset_parameters()
        self.network.reset_parameters()
        self.fc.reset_parameters()
