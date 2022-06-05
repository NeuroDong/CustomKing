import torch.nn as nn
from einops.layers.torch import Rearrange,Reduce
import torch

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

        weight = [[[[1,1,-1],[-1,0,1],[-1,-1,0]],[[-1,0,-1],[0,0,-1],[1,-1,0]],[[0,1,0],[1,0,1],[0,-1,1]]],
                [[[-1,-1,0],[-1,1,0],[-1,1,0]],[[1,-1,0],[-1,0,-1],[-1,0,0]],[[-1,0,1],[1,0,1],[0,-1,0]]]]
        weight = nn.Parameter(torch.Tensor(weight).reshape(2,27))

        for v_name,v in self.full_connect1.named_parameters():
            if v_name == "weight":
                print(self.full_connect1.weight)
                self.full_connect1.weight = weight
            if v_name == "bias":
                print(self.full_connect1.bias)
                self.full_connect1.bias = nn.Parameter(torch.Tensor([1.,0.]))

        #张量变换
        self.tensorTransform1 = nn.Sequential(Rearrange('b n c h w -> b n (c h w)'))

        # self.conv = nn.Conv2d(in_channel,out_channel,kernel_size[0],stride=stride,padding=padding,bias=bias)

        # weight = [[[[1,1,-1],[-1,0,1],[-1,-1,0]],[[-1,0,-1],[0,0,-1],[1,-1,0]],[[0,1,0],[1,0,1],[0,-1,1]]],
        #         [[[-1,-1,0],[-1,1,0],[-1,1,0]],[[1,-1,0],[-1,0,-1],[-1,0,0]],[[-1,0,1],[1,0,1],[0,-1,0]]]]
        # weight = nn.Parameter(torch.Tensor(weight))

        # for v_name,v in self.conv.named_parameters():
        #     if v_name == "weight":
        #         print(self.conv.weight)
        #         self.conv.weight = weight
        #     if v_name == "bias":
        #         print(self.conv.bias)
        #         self.conv.bias = nn.Parameter(torch.Tensor([1.,0.]))

        # for v in self.conv.parameters():
        #     print(v)

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

        vector_tmp = torch.einsum('...ij->...ji',vector_tmp)
        assert H_out*W_out == h,"reshape不对"
        vector_tmp = vector_tmp.reshape(vector_tmp.shape[0],vector_tmp.shape[1],H_out,W_out)
        # vector_tmp = self.conv(x)
        return vector_tmp

a = [[[0,1,1,2,2],[0,1,1,0,0],[1,1,0,1,0],[1,0,1,1,1],[0,2,0,1,0]],[[1,1,1,2,0],[0,2,1,1,2],[1,2,0,0,2],[0,2,1,2,1],[2,0,1,2,0]],
    [[2,0,2,0,2],[0,0,1,2,1],[1,0,2,2,1],[2,0,2,0,0],[0,0,1,1,2]]]
a = torch.Tensor(a).unsqueeze(dim=0).cuda()

conv = Sliding_Full_connection(3,2,(3,3),2,1).cuda()

out = conv(a)
print(out)



