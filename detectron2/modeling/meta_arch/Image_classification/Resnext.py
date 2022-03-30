from ..build import META_ARCH_REGISTRY
import torch
import torch.nn as nn

num_class = 5
resnext50_32x4d_params = [3, 4, 6, 3]
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


    def forward(self, data):
        #------------------预处理(data里面既含有image、label、width、height信息。)-----------------#
        batchsize = len(data)
        batch_images = []
        batch_label = []
        for i in range(0,batchsize,1):
            batch_images.append(data[i]["image"])
            batch_label.append(int(float(data[i]["y"])))
        batch_images=[image.tolist() for image in batch_images]
        batch_images_tensor = torch.tensor(batch_images,dtype=torch.float).cuda()


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

        x = self.avgpool(x) # torch.Size([1, 2048, 1, 1]) / torch.Size([1, 512])
        x = x.view(x.size(0), -1)   # torch.Size([1, 2048]) / torch.Size([1, 512])
        x = self.fc(x)      # torch.Size([1, 5])

        if self.training:
            #得到损失函数值
            batch_label = torch.tensor(batch_label,dtype=float).cuda()
            loss_fun = nn.CrossEntropyLoss()
            loss = loss_fun(x,batch_label.long())
            return loss
        else:
            #直接返回推理结果
            return x

@META_ARCH_REGISTRY.register()
def ResNeXt50():
    return ResNet(resnext50_32x4d_params, ResNeXtBlock)

@META_ARCH_REGISTRY.register()
def ResNeXt101():
    return ResNet(resnext101_32x8d_params, ResNeXtBlock)


if __name__ =='__main__':
    # model = ResNeXtBlock(in_places=256, places=128)
    # print(model)
    # model = ResNeXt50_32x4d()
    model = ResNeXt101()

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)
