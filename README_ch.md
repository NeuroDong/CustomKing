[English](https://github.com/dongdongdong1217/Detectron2-All/blob/main/README.md) | 简体中文

![Image text](https://github.com/dongdongdong1217/Detectron2-All/blob/main/docs/NeuroDong3.jpg)

# Detectron2-All是什么?
  Detectron2-All是一个基于Detectron2的神经网络快速搭建，其与Detectron2的区别在于：Detectron2只内置了目标检测，图像分割等领域的算法和数据集，面向的领域比较窄，而Detectron2-All致力于内置所有常用机器学习算法（以深度学习为主）和所有常用数据集，包括但不限于分类、回归、小样本、元学习等领域。
  
# 运行环境安装教程
见Detectron2的安装教程：https://detectron2.readthedocs.io/en/latest/tutorials/install.html。
环境如下：

Ubuntu20.04

CUDA10.1

Pytorch1.8.1

# 使用教程
## 使用内置的算法和数据集
所有内置算法、数据集和预训练权重都可以通过运行以下代码进行训练或评估，只需相应地更改参数即可。有关参数的选择，请见：https://github.com/dongdongdong1217/Detectron2-FC/tree/main/tools#readme.
```java  
  python3 tool/mian.py
```
