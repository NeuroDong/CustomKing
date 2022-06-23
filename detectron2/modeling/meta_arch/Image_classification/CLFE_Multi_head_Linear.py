import math
from turtle import forward
from unittest import result
import torch.nn as nn
import torch
from ..build import META_ARCH_REGISTRY

class CLFE_block(nn.Module):
    def __init__(self,image_size):
        super().__init__()
        self.lin1 = torch.nn.Linear(image_size[0]*image_size[1],1)
        self.fun = nn.ReLU(inplace=True)
        #self.W = torch.randn((512,224,224), requires_grad=True).cuda()
        self.W = nn.Linear(image_size[0],image_size[1],bias=False)
        self.bn = nn.BatchNorm2d(512)

    def forward(self,x):
        #--------------特征注意模块(CSFE)-------------#
        batchsize = x.shape[0]
        original_x = x
        #通道注意力
        x = torch.reshape(x,(batchsize,x.shape[1],x.shape[2]*x.shape[3]))
        x = self.lin1(x)
        x = self.fun(x)
        x = torch.reshape(x,(x.shape[0],x.shape[1],1,1))
        x = x * original_x 
        #x = self.fun(x)
        #x=self.W(x)
        #x = torch.sum(x,dim=1)
        #x = self.fun(x)
        #x = torch.reshape(x,(x.shape[0],1,x.shape[1],x.shape[2]))
        #x = original_x + x
        return self.fun(self.bn(x))

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, Q, K, V,d_k):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k) # scores : [batch_size, n_heads, len_q, len_k]        
        attn = self.softmax(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model=512,d_k=64,d_v=64,n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.d_k = torch.tensor(d_k)
        self.d_v = d_v
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.ScaledDotProductAttention = ScaledDotProductAttention()
    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = self.ScaledDotProductAttention(Q, K, V, self.d_k)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        return self.norm(output + residual)

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model=512,d_ff=2048):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.norm = nn.LayerNorm(d_model)
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs 
        output = self.fc(inputs) 
        return self.norm(output + residual)

class CLFE_and_Conv(nn.Module):
    def __init__(self,image_size):
        super().__init__()
        if isinstance(image_size,int):
            image_size = [image_size,image_size]

        self.conv1_1 = nn.Conv2d(3,256,3,1,1)
        self.conv1_2 = nn.Conv2d(3,256,11,1,5)
        self.bn1 = nn.BatchNorm2d(512)
        self.fun1 = nn.ReLU(inplace=True)
        self.CLFE1 = CLFE_block(image_size)

        self.conv2_1 = nn.Conv2d(512,256,3,2)
        self.conv2_2 = nn.Conv2d(512,256,7,2,2)
        self.fun2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(512)
        if image_size[0] % 2 ==0:
            conv_size = int((image_size[0]-2)/2) #image_size[0]是偶数
        else:
            conv_size = int((image_size[0]-2+1)/2) #image_size[0]是奇数
        self.CLFE2 = CLFE_block(image_size=(conv_size,conv_size))

        self.conv3_1 = nn.Conv2d(512,256,3,2)
        self.conv3_2 = nn.Conv2d(512,256,5,2,1)
        self.fun3 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(512)
        if conv_size % 2 ==0:
            conv_size = int((conv_size-2)/2) #image_size[0]是偶数
        else:
            conv_size = int((conv_size-2+1)/2) #image_size[0]是奇数
        self.CLFE3 = CLFE_block(image_size=(conv_size,conv_size))

        self.conv4_1 = nn.Conv2d(512,512,3,2,1)
        self.fun4 = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(512)
        if conv_size % 2 ==0:
            conv_size = int((conv_size+2-2)/2) #image_size[0]是偶数
        else:
            conv_size = int((conv_size+2-2+1)/2) #image_size[0]是偶数
        self.CLFE4 = CLFE_block(image_size=(conv_size,conv_size))

        self.conv5 = nn.Conv2d(512,512,3,2)
        self.bn5 = nn.BatchNorm2d(512)
        self.fun5 = nn.ReLU(inplace=True)
    def forward(self,x):
        x1_1 = self.conv1_1(x)
        x1_2 = self.conv1_2(x)
        x = torch.cat([x1_1, x1_2], dim=1)
        x = self.fun1(self.bn1(x))
        x = self.CLFE1(x)
        
        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)
        x = torch.cat([x2_1, x2_2], dim=1)
        x = self.fun2(self.bn2(x))
        x = self.CLFE2(x)

        x3_1 = self.conv3_1(x)
        x3_2 = self.conv3_2(x)
        x = torch.cat([x3_1, x3_2], dim=1)
        x = self.fun3(self.bn3(x))
        x = self.CLFE3(x)

        x4_1 = self.conv4_1(x)
        x = self.fun4(self.bn4(x4_1))
        x = self.CLFE4(x)

        x = self.conv5(x)
        x = self.fun5(self.bn5(x))
        return x

class Multi_Heads(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.multi_head1 = MultiHeadAttention()
        self.multi_head2 = MultiHeadAttention()
        self.Feed_forward = PoswiseFeedForwardNet()
        self.projection = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, cfg.num_classes)
        )
    def forward(self,x):
        x_shape = x.shape
        x = x.reshape(x_shape[0],x_shape[1],x_shape[2]*x_shape[3])
        x = x.permute(0,2,1)
        #-------------多头注意力强化特征-------------#
        x = self.multi_head1(x,x,x)
        x = self.multi_head2(x,x,x)

        x = self.Feed_forward(x)
        x = x.mean(dim = 1)
        x = self.projection(x)
        return x


@META_ARCH_REGISTRY.register()
class CLFE_Multi_head(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        if isinstance(cfg.ImageSize,int):
            cfg.ImageSize = (cfg.ImageSize,cfg.ImageSize)

        self.features = CLFE_and_Conv(cfg.ImageSize)
        self.classifier = Multi_Heads(cfg)
        

    def forward(self,batch_images_tensor):

        #----------------通道局部特征提取模块----------------#
        x = self.features(batch_images_tensor)
        
        #----------------多头注意力加分类器----------------#
        x = x.float()
        x = self.classifier(x)

        return x


