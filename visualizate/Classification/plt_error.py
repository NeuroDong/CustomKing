#------------------------------------------------------#
#此程序用来可视化训练过程中训练准确度和测试准确度
#------------------------------------------------------#

from turtle import color
import matplotlib.pyplot as plt
from numpy import arange
import json
import numpy as np
from torch import rand

def smooth_loss(file):
    
    data = []
    with open(file) as f:
        for line in f:
            data.append(json.loads(line))

    train_x = arange(0,len(data),1)*20
    train_list = [data[0]["train_acc"]]
    test_x = []
    test_list = []
    for i in data[1:]:
        train_list.append(i["train_acc"])
        if 'test_acc' in i.keys():
            test_x.append(i["iteration"])
            test_list.append(i["test_acc"])

    return train_x, train_list, test_x, test_list

def look_loss(filelist):
    plt.title('Error',fontsize=20)
    plt.xlabel('iter',fontsize=20)  # x轴标题
    plt.ylabel('Error(%)',fontsize=20)  # y轴标题

    #设置坐标刻度字体大小
    
    plt.ylim(0,100)
    plt.xticks(fontsize=20)
    plt.yticks([0,10,20,30,40,50,60,70,80,90,100],fontsize=20)

    legendlist = []
    n = 1
    for file in filelist:
        color = plt.cm.get_cmap("hsv", 5*n)
        train_x, train_list, test_x, test_list= smooth_loss(file) 
        train_list = (1-np.array(train_list))*100
        test_list = (1-np.array(test_list))*100
        plt.plot(train_x, train_list,linewidth = 3, linestyle='--', color=color(1))  # 绘制折线图，添加数据点，设置点的大小
        plt.plot(test_x, test_list, linewidth = 6, linestyle='-', color=color(1))
        file_name = file.split("/")
        legendlist.append(file_name[-2]+"_trainAcc")
        legendlist.append(file_name[-2]+"_testAcc")
    
    plt.legend(legendlist,fontsize=20)  # 设置折线名称
    plt.hlines([10], 0, train_x[-1], colors='b', linestyles='--', label='水平线')
    plt.xlim(0,train_x[-1])
    plt.show()  # 显示折线图

if __name__=="__main__":
    filelist = ["output/Cifar10/Resnet18_64000/metrics.json"]
    look_loss(filelist)