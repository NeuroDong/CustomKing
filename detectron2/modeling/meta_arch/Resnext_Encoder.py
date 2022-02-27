'''
New for ResNeXt:
1. Wider bottleneck
2. Add group for conv2
'''
from cgi import print_directory
import torch.nn as nn
import math
import torch
import numpy as np
from .build import META_ARCH_REGISTRY

num_class = 5
resnext101_32x8d_params = [3, 4, 23, 3]

# 定义Conv1层
def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class ResNeXtBlock(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 2, cardinality=32):
        super(ResNeXtBlock,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        # torch.Size([1, 256, 56, 56])
        # torch.Size([1, 512, 28, 28])
        # torch.Size([1, 1024, 14, 14])
        # torch.Size([1, 2048, 7, 7])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False, groups=cardinality),  # 使用了组卷积
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self,blocks, blockkinds, num_classes=num_class):
        super(ResNet,self).__init__()

        self.blockkinds = blockkinds
        self.conv1 = Conv1(in_planes = 3, places= 64)

        if self.blockkinds == ResNeXtBlock:
            self.expansion = 2
            # 64 -> 128
            self.layer1 = self.make_layer(in_places=64, places=128, block=blocks[0], stride=1)
            # 256 -> 256
            self.layer2 = self.make_layer(in_places=256, places=256, block=blocks[1], stride=2)
            # 512 -> 512
            self.layer3 = self.make_layer(in_places=512, places=512, block=blocks[2], stride=2)
            # 1024 -> 1024
            self.layer4 = self.make_layer(in_places=1024, places=1024, block=blocks[3], stride=2)

            self.fc = nn.Linear(2048, num_classes)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        # #自己加的
        self.layer5 = nn.Conv2d(2048,512,(1,1),1,0)

        # 初始化网络结构
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 采用了何凯明的初始化方法
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):

        layers = []
        # torch.Size([1, 256, 56, 56])
        layers.append(self.blockkinds(in_places, places, stride, downsampling =True))
        for i in range(1, block):
            layers.append(self.blockkinds(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, batch_images_tensor):

        # conv1层
        x = self.conv1(batch_images_tensor)   # torch.Size([1, 64, 56, 56])

        # conv2_x层
        x = self.layer1(x)  # torch.Size([1, 256, 56, 56])
        # conv3_x层
        x = self.layer2(x)  # torch.Size([1, 512, 28, 28])
        # conv4_x层
        x = self.layer3(x)  # torch.Size([1, 1024, 14, 14])
        # conv5_x层
        x = self.layer4(x)  # torch.Size([1, 2048, 7, 7])

        #x = self.avgpool(x) # torch.Size([1, 2048, 1, 1]) / torch.Size([1, 512])
        #x = x.view(x.size(0), -1)   # torch.Size([1, 2048]) / torch.Size([1, 512])
        #x = self.fc(x)      # torch.Size([1, 5])
        x = self.layer5(x)
        x = x.reshape(len(batch_images_tensor),49,512)

        return x

##############################################################
# Transformer Parameters
d_model = 512  # Embedding Size
d_ff = 2048 # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention


#位置编码函数
def get_sinusoid_encoding_table(n_position, d_model): 
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)

#Pad Mask
def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

#Subsequence Mask
def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte().cuda().long()
    return subsequence_mask

#ScaledDotProductAttention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn

#MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        return self.norm(output + residual), attn

# #MultiHeadAttention
# class MultiHeadAttention_fuse(nn.Module):
#     def __init__(self):
#         super(MultiHeadAttention_fuse, self).__init__()
#         self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
#         self.W_K = nn.Linear(512, d_k * n_heads, bias=False)
#         self.W_V = nn.Linear(512, d_v * n_heads, bias=False)
#         self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
#         self.norm = nn.LayerNorm(d_model)
#     def forward(self, input_Q, input_K, input_V, attn_mask):
#         '''
#         input_Q: [batch_size, len_q, d_model]
#         input_K: [batch_size, len_k, d_model]
#         input_V: [batch_size, len_v(=len_k), d_model]
#         attn_mask: [batch_size, seq_len, seq_len]
#         '''
#         residual, batch_size = input_Q, input_Q.size(0)
#         add_mask = torch.zeros((batch_size,1,1)).bool().cuda()
#         attn_mask = torch.cat((attn_mask,add_mask),dim=2)
#         # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
#         Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
#         K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
#         V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

#         attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

#         # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
#         context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
#         context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
#         output = self.fc(context) # [batch_size, len_q, d_model]
#         return self.norm(output + residual), attn

#FeedForward Layer
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
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
        return self.norm(output + residual) # [batch_size, seq_len, d_model]

#Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        

        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

#Encoder,src_vocab_size代表输入字典的长度
class Encoder(nn.Module):
    def __init__(self,src_vocab_size):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model) #得到词嵌入
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_vocab_size, d_model),freeze=True) #位置编码得到位置嵌入
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs,image_output):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        word_emb = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        pos_emb = self.pos_emb(enc_inputs) # [batch_size, src_len, d_model]
        enc_outputs = word_emb + pos_emb

        enc_outputs = torch.cat((enc_outputs,image_output),dim=1) #融合过程数据和图像数据
        
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]
        batch_size = enc_inputs.size(0)
        add_mask = torch.zeros((batch_size,35,49)).bool().cuda()
        enc_self_attn_mask = torch.cat((enc_self_attn_mask,add_mask),dim=2)
        add_mask = torch.zeros((batch_size,49,84)).bool().cuda()
        enc_self_attn_mask = torch.cat((enc_self_attn_mask,add_mask),dim=1)
        
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

#Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs) # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn

#Decoder
class Decoder(nn.Module):
    def __init__(self,tgt_vocab_size):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_vocab_size, d_model),freeze=True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        '''
        word_emb = self.tgt_emb(dec_inputs) # [batch_size, tgt_len, d_model]
        pos_emb = self.pos_emb(dec_inputs) # [batch_size, tgt_len, d_model]
        dec_outputs = word_emb + pos_emb
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs) # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequent_mask = get_attn_subsequence_mask(dec_inputs) # [batch_size, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0) # [batch_size, tgt_len, tgt_len]

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs) # [batc_size, tgt_len, src_len]
        batch_size = enc_inputs.size(0)
        add_mask = torch.zeros((batch_size,1,49)).bool().cuda()
        dec_enc_attn_mask = torch.cat((dec_enc_attn_mask,add_mask),dim=2)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

@META_ARCH_REGISTRY.register()
class Resnext_encoder(nn.Module):
    def __init__(self,cfg):
        super(Resnext_encoder, self).__init__()
        self.image_network = ResNet(resnext101_32x8d_params, ResNeXtBlock).cuda()  #resnext101
        self.encoder = Encoder(cfg.src_vocab_size)

        self.projection = nn.Linear(d_model,cfg.tgt_vocab_size, bias=False)
        self.loss_fun = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self,data):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''

        #------------------获取并预处理图像数据(data里面既含有image、label、width、height信息)---#
        batchsize = len(data)
        batch_images = []
        for i in range(0,batchsize,1):
            batch_images.append(data[i]["image"])
        batch_images=[image.tolist() for image in batch_images]
        batch_images_tensor = torch.tensor(batch_images,dtype=torch.float).cuda()

        #------------------获取并预处理过程数据(data里面既含有image_name、x、y信息。)------------#
        batch_x = []
        batch_y = []
        for i in range(0,batchsize,1):
            batch_x.append(data[i]["x"])
            batch_y.append([int(float(data[i]["y"]))])
        batch_x=[x.tolist() for x in batch_x]
        enc_inputs = torch.tensor(batch_x,dtype=torch.float).cuda().long()
        dec_inputs = torch.tensor(batch_y ,dtype=torch.float).cuda().long()

        #----------------------------网络向前推理------------------------------------#
        #-----------------推理SE-Resnext------------#
        image_output = self.image_network(batch_images_tensor)

        #----------------推理Transformer------------#
        enc_outputs, enc_self_attns = self.encoder(enc_inputs,image_output)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        enc_outputs = enc_outputs.mean(dim=1)
        
        #----------------------------解码生成损失函数---------------------------------#
        dec_logits = self.projection(enc_outputs) # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        outputs, enc_self_attns = dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns
        if self.training:
            loss = self.loss_fun(outputs, dec_inputs.view(-1))    
            return loss
        else:
            return outputs

if __name__=="__main__":
    model = Resnext_encoder()
    output = model()
    print("output:",output.shape)