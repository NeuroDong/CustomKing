from turtle import forward
import torch.nn as nn
import torch
from einops.layers.torch import Rearrange,Reduce
from ..build import META_ARCH_REGISTRY
import numpy as np
import time

class ScaledDotProductAttention(nn.Module):
    def __init__(self,d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : [batch_size, n_heads, len_q, len_k]        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context

class MLPBlock(nn.Sequential):
    """Transformer MLP block."""

    def __init__(self, in_dim: int, mlp_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, mlp_dim)
        self.act = nn.GELU()
        #self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(mlp_dim, in_dim)
        #self.dropout_2 = nn.Dropout(dropout)

class Sliding_attention(nn.Module):
    def __init__(self,image_size,in_channel,out_channel,kernel_size,stride: int = 1,pad: int=0):
        super().__init__()
        if isinstance(image_size,int):
            image_size = [image_size,image_size]

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.H_out = (image_size[0]+2*pad-kernel_size[0])//stride +1
        self.W_out = (image_size[1]+2*pad-kernel_size[1])//stride +1
        self.stride = stride
        self.pad = pad
        
        self.to_k = nn.Sequential(nn.Linear(in_channel*kernel_size[0]*kernel_size[1],out_channel),nn.LayerNorm(out_channel))
        self.to_q = nn.Sequential(nn.Linear(in_channel*kernel_size[0]*kernel_size[1],out_channel),nn.LayerNorm(out_channel))
        self.to_v = nn.Sequential(nn.Linear(in_channel*kernel_size[0]*kernel_size[1],out_channel),nn.LayerNorm(out_channel))
        self.ScaledDotProductAttention = ScaledDotProductAttention(out_channel)
        self.norm_fun1 = nn.Sequential(nn.LayerNorm(out_channel),nn.GELU())
        #self.mlp = nn.Sequential(MLPBlock(out_channel,out_channel),nn.LayerNorm(out_channel))

        self.channel_attention = nn.Linear(self.H_out*self.W_out,1)
        self.norm_fun2 = nn.Sequential(nn.BatchNorm2d(out_channel),nn.GELU())

        #张量变换
        self.tensorTransform1 = nn.Sequential(Rearrange('b n c h w -> b n (c h w)'))

    def forward(self,x):
        x_pad = torch.zeros(x.shape[0],x.shape[1],x.shape[2]+2*self.pad,x.shape[3]+2*self.pad,requires_grad=False)
        x_pad[:,:,self.pad:self.pad+x.shape[2],self.pad:self.pad+x.shape[3]] = x
        vector_tensor = torch.zeros(x_pad.shape[0],self.H_out*self.W_out,x_pad.shape[1],self.kernel_size[0],self.kernel_size[1])
        h = 0
        for i in range(0,x_pad.shape[2]-self.kernel_size[0] + 1,self.stride):
            for j in range(0,x_pad.shape[3]-self.kernel_size[1] + 1,self.stride):
                vector_tensor[:,h] = x_pad[:,:,i:i+self.kernel_size[0],j:j+self.kernel_size[1]]
                h += 1

        vector_tensor = self.tensorTransform1(vector_tensor).cuda().clone().detach()
        k = self.to_k(vector_tensor)
        q = self.to_q(vector_tensor)
        v = self.to_v(vector_tensor)
        vector_tmp = self.ScaledDotProductAttention(q,k,v)
        vector_tmp = self.norm_fun1(vector_tmp)
        #vector_tmp = self.mlp(vector_tmp)

        vector_tmp = torch.einsum('...ij->...ji',vector_tmp)
        vector_tmp = self.channel_attention(vector_tmp) * vector_tmp
        assert self.H_out*self.W_out == h,"reshape不对"
        vector_tmp = vector_tmp.reshape(vector_tmp.shape[0],vector_tmp.shape[1],self.H_out,self.W_out)
        return self.norm_fun2(vector_tmp)

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model=512,d_ff=2048):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
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

@META_ARCH_REGISTRY.register()
class Sliding_attention_Network(nn.Module):
    def __init__(self,cfg):
        super().__init__()

        image_size=cfg.ImageSize
        if isinstance(image_size,int):
            image_size = [image_size,image_size]

        self.Sliding_attention1 = Sliding_attention(image_size=image_size,in_channel=3,out_channel=64,kernel_size=(7,7),stride=2,pad=3)
        image_size[0] = (image_size[0]+2*3-7)//2 +1 #(image_size[0]+2*pad-kernel_size[0])//stride +1
        image_size[1] = (image_size[1]+2*3-7)//2 +1

        self.Sliding_attention2 = nn.Sequential(
            Sliding_attention(image_size=image_size,in_channel=64,out_channel=128,kernel_size=(1,1),stride=1),
            Sliding_attention(image_size=image_size,in_channel=128,out_channel=128,kernel_size=(3,3),stride=2,pad=1),
            Sliding_attention(image_size=[(image_size[0]+2*1-3)//2 +1,(image_size[0]+2*1-3)//2 +1],in_channel=128,out_channel=256,kernel_size=(1,1),stride=1))
        self.shotcut2 = Sliding_attention(image_size=image_size,in_channel=64,out_channel=256,kernel_size=(1,1),stride=2)
        self.norm2 = nn.BatchNorm2d(256)
        image_size[0] = (image_size[0]+2*1-3)//2 +1
        image_size[1] = (image_size[1]+2*1-3)//2 +1

        self.Sliding_attention3 = nn.Sequential(
            Sliding_attention(image_size=image_size,in_channel=256,out_channel=256,kernel_size=(1,1),stride=1),
            Sliding_attention(image_size=image_size,in_channel=256,out_channel=256,kernel_size=(3,3),stride=2,pad=1),
            Sliding_attention(image_size=[(image_size[0]+2*1-3)//2 +1,(image_size[0]+2*1-3)//2 +1],in_channel=256,out_channel=512,kernel_size=(1,1),stride=1))
        self.shotcut3 = Sliding_attention(image_size=image_size,in_channel=256,out_channel=512,kernel_size=(1,1),stride=2)
        self.norm3 = nn.BatchNorm2d(512)
        image_size[0] = (image_size[0]+2*1-3)//2 +1
        image_size[1] = (image_size[1]+2*1-3)//2 +1

        self.Sliding_attention4 = nn.Sequential(
            Sliding_attention(image_size=image_size,in_channel=512,out_channel=512,kernel_size=(1,1),stride=1),
            Sliding_attention(image_size=image_size,in_channel=512,out_channel=512,kernel_size=(3,3),stride=2,pad=1),
            Sliding_attention(image_size=[(image_size[0]+2*1-3)//2 +1,(image_size[0]+2*1-3)//2 +1],in_channel=512,out_channel=1024,kernel_size=(1,1),stride=1))
        self.shotcut4 = Sliding_attention(image_size=image_size,in_channel=512,out_channel=1024,kernel_size=(1,1),stride=2)
        self.norm4 = nn.BatchNorm2d(1024)
        image_size[0] = (image_size[0]+2*1-3)//2 +1
        image_size[1] = (image_size[1]+2*1-3)//2 +1

        # self.Sliding_attention5 = nn.Sequential(
        #     Sliding_attention(image_size=image_size,in_channel=1024,out_channel=1024,kernel_size=(1,1),stride=1),
        #     Sliding_attention(image_size=image_size,in_channel=1024,out_channel=1024,kernel_size=(3,3),stride=2,pad=1),
        #     Sliding_attention(image_size=[(image_size[0]+2*1-3)//2 +1,(image_size[0]+2*1-3)//2 +1],in_channel=1024,out_channel=2048,kernel_size=(1,1),stride=1))
        # self.shotcut5 = Sliding_attention(image_size=image_size,in_channel=1024,out_channel=2048,kernel_size=(1,1),stride=2)
        # self.norm5 = nn.BatchNorm2d(2048)
        # image_size[0] = (image_size[0]+2*1-3)//2 +1
        # image_size[1] = (image_size[1]+2*1-3)//2 +1

        self.FeedForward = nn.Sequential(Rearrange('b c h w -> b c (h w)'),
                            Rearrange('b c n -> b n c'),
                            PoswiseFeedForwardNet(1024,2048),
                            Reduce('b n c -> b c','mean'))

        self.projection = nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024, cfg.num_classes))
    
    def forward(self,batch_images_tensor):

        #推理部分
        #time1 = time.time()
        out = self.Sliding_attention1(batch_images_tensor)
        out = self.norm2(self.Sliding_attention2(out) + self.shotcut2(out))
        out = self.norm3(self.Sliding_attention3(out)+self.shotcut3(out))
        out = self.norm4(self.Sliding_attention4(out)+self.shotcut4(out))
        # out = self.norm5(self.Sliding_attention5(out)+self.shotcut5(out))
        out = self.FeedForward(out)
        out = self.projection(out)
        #print("用时：",time.time()-time1)

        return out


        








