#------------------------------------------------------#
#此程序用来对损失函数进行平滑，并将损失值用plt画出来
#使用的时候修改保存损失值文件的路径即可
#------------------------------------------------------#

import matplotlib.pyplot as plt
from numpy import arange
import json

filelist = ["output/Cifar10/Resnet_data_enhancement_nopool/metrics.json","output/Cifar10/Complete_graph_network18_data_enhancement_no_pool/metrics.json"]

def smooth_loss(file,weight=0.95):
    
    data = []
    with open(file) as f:
        for line in f:
            data.append(json.loads(line))

    x = arange(0,len(data),1)*20
    loss_list = [data[0]["loss"]]
    for i in data[1:]:
        loss_list.append(loss_list[-1] * weight + (1-weight)*i["loss"])

    return x,loss_list

def look_loss(filelist):
    plt.title('Train_loss')
    plt.xlabel('step')  # x轴标题
    plt.ylabel('loss')  # y轴标题

    legendlist = []
    h = 1
    for file in filelist:
        x,loss_list= smooth_loss(file)
        plt.plot(x, loss_list, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
        file_name = file.split("/")
        #legendlist.append("file" + str(h))
        legendlist.append(file_name[-2])
        h = h + 1
    
    plt.legend(legendlist)  # 设置折线名称
    plt.show()  # 显示折线图

if __name__=="__main__":
    look_loss(filelist)