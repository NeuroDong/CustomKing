from cgi import print_environ
from operator import mod
from random import sample
from tabnanny import verbose
from tkinter import SEL_LAST
from turtle import forward
from typing_extensions import Required
from cv2 import sepFilter2D
import torch.nn as nn
import torch
import numpy as np
from ..build import META_ARCH_REGISTRY
import time
import copy

class LT_unit(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,stride,pad):
        super(LT_unit,self).__init__()

        # self.linear_list = nn.Parameter(torch.randn(in_channel,kernel_size*kernel_size,out_channel),requires_grad=True)
        lin_list = []
        for i in range(0,in_channel,1):
            lin_list.append(nn.Linear(kernel_size*kernel_size,out_channel))
        self.linear_list = nn.ModuleList(lin_list)
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.size = kernel_size
        self.stride = stride
        self.pad = pad
        self.fun1 = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channel)

    def forward(self,x):
        x_pad = torch.zeros(x.shape[0],x.shape[1],x.shape[2]+2*self.pad,x.shape[3]+2*self.pad,requires_grad=False).cuda()
        x_pad[:,:,self.pad:self.pad+x.shape[2],self.pad:self.pad+x.shape[3]] = x
        vector_tensor = torch.zeros(x_pad.shape[0],x_pad.shape[1],(x_pad.shape[3]-self.size+1)*(x_pad.shape[2]-self.size+1),self.size*self.size,requires_grad=False).cuda()
        h = 0
        for i in range(0,x_pad.shape[3]-self.size + 1,self.stride):
            for j in range(0,x_pad.shape[2]-self.size + 1,self.stride):
                tmp = x_pad[:,:,i:i+self.size,j:j+self.size]
                tmp = tmp.reshape(x_pad.shape[0],x_pad.shape[1],self.size*self.size)
                vector_tensor[:,:,h,:] = tmp.clone().detach()
                h += 1
        
        # out_tmp = torch.einsum("bikj,ijt->bkt",vector_tensor,self.linear_list)
        vector_tensor = vector_tensor
        out_tmp = self.linear_list[0](vector_tensor[:,0])
        for k in range(1,self.in_channel,1):
            out_tmp += self.linear_list[k](vector_tensor[:,k])
        
        out_tmp = out_tmp.permute(0,2,1)
        vector_tensor = out_tmp.reshape(out_tmp.shape[0],out_tmp.shape[1],int(out_tmp.shape[2]**0.5),int(out_tmp.shape[2]**0.5))
        return self.fun1(self.norm(vector_tensor))

class LT_block(nn.Module):
    def __init__(self):
        super().__init__()
        #用3个3*3代替一个7*7
        self.LT_unit1 = nn.Sequential(LT_unit(3,64,3,1,1),LT_unit(64,128,3,2,1),LT_unit(128,256,3,1,1))
        self.shotcut1 = nn.Sequential(LT_unit(3,256,1,2,0))
        self.fun1 = nn.ReLU()

        #用2个3*3代替一个5*5
        self.LT_unit2 = nn.Sequential(LT_unit(256,512,3,1,1),LT_unit(512,512,3,2,1))
        self.shotcut2 = LT_unit(256,512,1,2,0)
        self.fun2 = nn.ReLU()
        
        self.LT_unit3 = LT_unit(512,512,3,2,1)

    def forward(self,x):
        #形成残差
        out = self.LT_unit1(x)
        x = self.fun1(self.shotcut1(x)+out)
        #形成残差
        out = self.LT_unit2(x)
        x = self.fun2(self.shotcut2(x)+out)

        x = self.LT_unit3(x)
        return x

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V,d_k):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, len_q, len_k]        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model=512,d_k=64,d_v=64,n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
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
        context = ScaledDotProductAttention()(Q, K, V, self.d_k)
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

class Multi_Heads(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.multi_head1 = MultiHeadAttention()
        self.multi_head2 = MultiHeadAttention()
        self.Feed_forward = PoswiseFeedForwardNet() 
        self.projection = nn.Sequential(nn.LayerNorm(512),nn.Linear(512, cfg.Arguments2))
    def forward(self,x):
        #-------------多头注意力强化特征-------------#
        x = self.multi_head1(x,x,x)
        x = self.multi_head2(x,x,x)
        x = self.Feed_forward(x)
        
        x = x.mean(dim = 1)
        x = self.projection(x)
        return x

@META_ARCH_REGISTRY.register()
class LT_FN(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.feature = LT_block()
        #self.Embedding = nn.Embedding(49,512)
        #self.Pos_Emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(49,512),freeze=True)
        #self.guodu = nn.Sequential(nn.Linear(49,512),nn.LayerNorm(512))
        self.classfier = Multi_Heads(cfg)
        #self.Feed_forward = PoswiseFeedForwardNet()
        #self.projection = nn.Sequential(nn.LayerNorm(512),nn.Linear(512,cfg.Arguments2))
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

        #-------------------推理--------------------------------#
        x = self.feature(batch_images_tensor)
        x = x.reshape(x.shape[0],x.shape[1],x.shape[2]*x.shape[3]).permute(0,2,1)
        #x = x.reshape(x.shape[0],x.shape[1]*x.shape[2],x.shape[3])
        #x = self.Embedding(x) + self.Pos_Emb(x)
        #x = self.guodu(x)
        x = self.classfier(x)
        #print(x.shape)
        #print(batch_label)
        #print(x.shape)
        #print(batch_label.shape)

        if self.training:
            #得到损失函数值
            batch_label = torch.tensor(batch_label,dtype=float).cuda()
            loss_fun = nn.CrossEntropyLoss()
            #print("x:",x)
            #print("batch_label:",batch_label.long())
            loss = loss_fun(x,batch_label.long())
            #print("loss:",loss)
            return loss
        else:
            #直接返回推理结果
            return x

if __name__=="__main__":
    input_data = torch.randn((4,3,32,32))
    print(input_data.shape)
    model = LT_block(3,10,7,512,2)
    print(model)
    out = model(input_data)
    print(out.shape)
    for n,v in model.named_parameters():
	    print(n+":",v.shape)