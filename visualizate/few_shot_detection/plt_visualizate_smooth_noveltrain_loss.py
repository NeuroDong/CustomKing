#------------------------------------------------------#
#此程序用来对损失函数进行平滑，并将损失值用plt画出来
#使用的时候修改保存损失值文件的路径即可
#------------------------------------------------------#

import matplotlib.pyplot as plt
from numpy import arange
import json
file = "checkpoints/voc/5shot_Seed0/R-101_9999/metrics.json"

def smooth_loss(weight=0.95):
    
    data = []
    with open(file) as f:
        for line in f:
            data.append(json.loads(line))

    x = arange(0,len(data),1)*20
    total_loss_list = [data[0]["total_loss"]]
    loss_box_reg_list = [data[0]["loss_box_reg"]]
    loss_cls_list = [data[0]["loss_cls"]]
    loss_rpn_cls_list = [data[0]["loss_rpn_cls"]]
    loss_rpn_loc_list = [data[0]["loss_rpn_loc"]]
    loss_contrast_list = [data[0]["loss_contrast"]]
    lr_list = [data[0]["lr"]]
    for i in data[1:]:
        total_loss_list.append(total_loss_list[-1] * weight + (1-weight)*i["total_loss"])
        loss_box_reg_list.append(loss_box_reg_list[-1] * weight + (1-weight)*i["loss_box_reg"])
        loss_cls_list.append(loss_cls_list[-1]* weight + (1-weight)*i["loss_cls"])
        loss_rpn_cls_list.append(loss_rpn_cls_list[-1] * weight + (1-weight)*i["loss_rpn_cls"])
        loss_rpn_loc_list.append(loss_rpn_loc_list[-1] * weight + (1-weight)*i["loss_rpn_loc"])
        loss_contrast_list.append(loss_contrast_list[-1] * weight + (1-weight)*i["loss_contrast"])
        lr_list.append(lr_list[-1] * weight + (1-weight)*i["lr"])

    return x,loss_rpn_loc_list,loss_rpn_cls_list,loss_box_reg_list,loss_cls_list,total_loss_list,loss_contrast_list,lr_list


def look_all_loss():
    x,rpn_loc_smoothed,rpn_cls_smoothed,roi_loc_smoothed,roi_cls_smoothed,total_loss_smoothed,loss_contrast_smoothed,lr_list= smooth_loss()

    plt.title('Train_loss')
    plt.xlabel('step')  # x轴标题
    plt.ylabel('loss')  # y轴标题

    plt.plot(x, rpn_loc_smoothed, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
    plt.plot(x, rpn_cls_smoothed, marker='o', markersize=3)
    plt.plot(x, roi_loc_smoothed, marker='o', markersize=3)
    plt.plot(x, roi_cls_smoothed, marker='o', markersize=3)
    plt.plot(x, total_loss_smoothed, marker='o', markersize=3)
    plt.plot(x,loss_contrast_smoothed,marker='o', markersize=3)
    plt.legend(['rpn_loc', 'rpn_cls', 'roi_loc', 'roi_cls', 'total_loss','loss_contrast'])  # 设置折线名称
    plt.show()  # 显示折线图


if __name__=="__main__":
    look_all_loss()