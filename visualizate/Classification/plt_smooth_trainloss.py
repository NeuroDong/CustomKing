#------------------------------------------------------#
#此程序用来对损失函数进行平滑，并将损失值用plt画出来
#使用的时候修改保存损失值文件的路径即可
#------------------------------------------------------#

import matplotlib.pyplot as plt
from numpy import arange
import json
file = "output/classification/icron_water/metrics.json"

def smooth_loss(weight=0.95):
    
    data = []
    with open(file) as f:
        for line in f:
            data.append(json.loads(line))

    x = arange(0,len(data),1)*20
    loss_list = [data[0]["loss"]]
    for i in data[1:]:
        loss_list.append(loss_list[-1] * weight + (1-weight)*i["loss"])

    return x,loss_list

def look_loss():
    x,loss_list= smooth_loss()

    plt.title('Train_loss')
    plt.xlabel('step')  # x轴标题
    plt.ylabel('loss')  # y轴标题

    plt.plot(x, loss_list, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
    plt.legend(["loss"])  # 设置折线名称
    plt.show()  # 显示折线图

if __name__=="__main__":
    look_loss()