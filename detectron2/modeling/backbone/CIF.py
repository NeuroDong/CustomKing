# The copyright belongs to Jinzong Dong, whose email address is dongjinzong@126.com.
import torch.nn as nn
import torch
import torchvision

from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    DeformConv,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)
from detectron2.modeling.backbone.backbone import Backbone

from .build import BACKBONE_REGISTRY

__all__ = ["build_cif_backbone"]

class CIF_block(nn.Module):
    '''
    The size of the input feature map of CIF_block and the size of the output fe
    '''
    def __init__(self,image_size):
        super().__init__()
        self.lin1 = torch.nn.Linear(image_size[0]*image_size[1],1)
        self.fun = nn.ReLU(inplace=True)
        #self.W = torch.randn((512,224,224), requires_grad=True).cuda()
        self.W = nn.Linear(image_size[1],image_size[1],bias=False)
        self.bn = nn.BatchNorm2d(512)

    def forward(self,x):
        #--------------特征注意模块(CSFE)-------------#
        batchsize = x.shape[0]
        original_x = x
        #通道注意力
        x = x.reshape(batchsize,x.shape[1],x.shape[2]*x.shape[3])

        x = self.lin1(x)
        x = self.fun(x)
        x = torch.reshape(x,(x.shape[0],x.shape[1],1,1))
        x = x * original_x 
        x = self.fun(x)
        x=self.W(x)
        x = torch.sum(x,dim=1)
        x = self.fun(x)
        x = torch.reshape(x,(x.shape[0],1,x.shape[1],x.shape[2]))
        x = original_x + x
        return self.fun(self.bn(x))

class CIF_and_Conv(Backbone):
    def __init__(self,image_size,input_channels=3):
        super().__init__()
        self.resize = torchvision.transforms.Resize(image_size)

        self.conv1_1 = nn.Conv2d(input_channels,256,3,1,1)
        self.conv1_2 = nn.Conv2d(input_channels,256,11,1,5)
        self.bn1 = nn.BatchNorm2d(512)
        self.fun1 = nn.ReLU(inplace=True)
        self.CLFE1 = CIF_block(image_size)

        self.conv2_1 = nn.Conv2d(512,256,3,2)
        self.conv2_2 = nn.Conv2d(512,256,7,2,2)
        self.fun2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(512)
        if image_size[0] % 2 ==0:
            conv_size0 = int((image_size[0]-2)/2) #image_size[0]是偶数
        else:
            conv_size0 = int((image_size[0]-2+1)/2) #image_size[0]是奇数
        if image_size[1] % 2 ==0:
            conv_size1 = int((image_size[1]-2)/2) #image_size[1]是偶数
        else:
            conv_size1 = int((image_size[1]-2+1)/2) #image_size[1]是奇数
        self.CLFE2 = CIF_block(image_size=(conv_size0,conv_size1))

        self.conv3_1 = nn.Conv2d(512,256,3,2)
        self.conv3_2 = nn.Conv2d(512,256,5,2,1)
        self.fun3 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(512)
        if conv_size0 % 2 ==0:
            conv_size0 = int((conv_size0-2)/2) #conv_size0是偶数
        else:
            conv_size0 = int((conv_size0-2+1)/2) #conv_size0是奇数
        if conv_size1 % 2 ==0:
            conv_size1 = int((conv_size1-2)/2) #conv_size1是偶数
        else:
            conv_size1 = int((conv_size1-2+1)/2) #conv_size1是奇数
        self.CLFE3 = CIF_block(image_size=(conv_size0,conv_size1))

        self.conv4_1 = nn.Conv2d(512,512,3,2,1)
        self.fun4 = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(512)
        if conv_size0 % 2 ==0:
            conv_size0 = int((conv_size0)/2) #conv_size0是偶数
        else:
            conv_size0 = int((conv_size0+1)/2) #conv_size0是奇数
        if conv_size1 % 2 ==0:
            conv_size1 = int((conv_size1)/2) #conv_size1是偶数
        else:
            conv_size1 = int((conv_size1+1)/2) #conv_size1是奇数
        self.CLFE4 = CIF_block(image_size=(conv_size0,conv_size1))

        self.conv5 = nn.Conv2d(512,512,3,2)
        self.bn5 = nn.BatchNorm2d(512)
        self.fun5 = nn.ReLU(inplace=True)

        self._out_feature_channel = 512
        self._out_feature_stride = 16


    def forward(self,x):
        x = self.resize(x)
        x1_1 = self.conv1_1(x)
        x1_2 = self.conv1_2(x)
        x = torch.cat([x1_1, x1_2], dim=1)
        x = self.fun1(self.bn1(x))
        self.CLFE1(x)
        
        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)
        x = torch.cat([x2_1, x2_2], dim=1)
        x = self.fun2(self.bn2(x))
        self.CLFE2(x)

        x3_1 = self.conv3_1(x)
        x3_2 = self.conv3_2(x)
        x = torch.cat([x3_1, x3_2], dim=1)
        x = self.fun3(self.bn3(x))
        self.CLFE3(x)

        x4_1 = self.conv4_1(x)
        x = self.fun4(self.bn4(x4_1))
        self.CLFE4(x)

        x = self.conv5(x)
        x = self.fun5(self.bn5(x))
        return x

    def output_shape(self):
        return {"CIF":ShapeSpec(channels=self._out_feature_channel, stride=self._out_feature_stride)}

@BACKBONE_REGISTRY.register()
def build_cif_backbone(cfg, input_shape):
    """
    Create a CIF instance from config.

    Returns:
        CIF: a :class:`CIF` instance.
    """
    return CIF_and_Conv(cfg.CIF.IMAGE.IMAGE_SIZE,input_channels=input_shape.channels)


    
