from unittest import result
import torch.nn as nn
import torch
import numpy as np
from ..build import META_ARCH_REGISTRY

class CLFE_block(nn.Module):
    def __init__(self,image_size=(224,224)):
        super().__init__()
        self.depthwise1 = nn.Conv2d(512,512,7,1,3,1,groups=512)
        self.lin1 = torch.nn.Linear(image_size[0]*image_size[1],1)
        self.fun = nn.ReLU(inplace=True)
        self.depthwise2 = nn.Conv2d(512,512,7,1,3,1,groups=512)
        self.bn = nn.BatchNorm2d(512)

    def forward(self,x):
        #--------------特征注意模块(CSFE)-------------#
        batchsize = x.shape[0]
        original_x = x
        
        x = self.depthwise1(x) #加强每个通道的区别和特征
        x = self.fun(x)

        #通道注意力
        x = torch.reshape(x,(batchsize,x.shape[1],x.shape[2]*x.shape[3]))
        x = self.lin1(x)
        x = self.fun(x)
        x = torch.reshape(x,(x.shape[0],x.shape[1],1,1))
        x = x * original_x 
        x = self.fun(x)
        #局部注意力
        x = self.depthwise2(x) #分别对每个通道提取局部特征
        x = self.fun(x)

        x = original_x + x
        return self.fun(self.bn(x))

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

@META_ARCH_REGISTRY.register()
class CLFE_Multi_head_Depthwise(nn.Module):
    def __init__(self,cfg,image_size=(224,224)):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3,256,3,1,1)
        self.conv1_2 = nn.Conv2d(3,256,11,1,5)
        self.bn1 = nn.BatchNorm2d(256)
        self.fun1 = nn.ReLU(inplace=True)

        self.CLFE1 = CLFE_block(image_size)
        #self.CLFE2 = CLFE_block(image_size)

        self.conv2 = nn.Conv2d(512,512,3,2)
        self.conv3 = nn.Conv2d(512,512,3,2)
        self.conv4 = nn.Conv2d(512,512,3,2)
        self.conv5 = nn.Conv2d(512,512,3,2)
        self.bn2 = nn.BatchNorm2d(512)
        

        self.multi_head1 = MultiHeadAttention()
        #self.multi_head2 = MultiHeadAttention()
        self.Feed_forward = PoswiseFeedForwardNet()

        self.projection = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, cfg.Arguments1)
        )

    def forward(self,data):

        #------------------预处理(data里面既含有image、label、width、height信息。)-----------------#
        batchsize = len(data)
        batch_images = []
        batch_label = []
        for i in range(0,batchsize,1):
            batch_images.append(data[i]["image"])
            batch_label.append(int(float(data[i]["y"])))
        batch_images=[image.tolist() for image in batch_images]
        batch_images_tensor = torch.tensor(batch_images,dtype=torch.float).cuda()

        batchsize = batch_images_tensor.shape[0]

        #----------------特征引入模块----------------#
        x1 = self.fun1(self.bn1(self.conv1_1(batch_images_tensor)))
        x2 = self.fun1(self.bn1(self.conv1_2(batch_images_tensor)))
        x = torch.cat([x1, x2], dim=1)
        
        #--------------特征注意模块(CSFE)-------------#
        x = self.CLFE1(x)
        #x = self.CLFE2(x)

        #-------------调整特征图大小----------------#
        x = self.fun1(self.bn2(self.conv2(x)))
        x = self.fun1(self.bn2(self.conv3(x)))
        x = self.fun1(self.bn2(self.conv4(x)))
        x = self.fun1(self.bn2(self.conv5(x)))
        x = x.reshape(batchsize,169,512)

        #-------------多头注意力强化特征-------------#
        x = self.multi_head1(x,x,x)
        #x = self.multi_head2(x,x,x)

        x = self.Feed_forward(x)
        x = x.mean(dim = 1)
        x = self.projection(x)

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

# model = CLFE_Multi_head().cuda()
# image = torch.ones((4,3,224,224)).cuda()
# result = model(image)
# print(result.shape)

