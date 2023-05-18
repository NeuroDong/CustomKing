[English](https://github.com/dongdongdong1217/Detectron2-All/blob/main/README.md) | 简体中文

# 1. CustonKing是什么?
CustomKing是一个机器学习的框架，是用Fvcore写的一个类似于Detectron2的框架。 CustomKing与Detectron2的区别如下：

(1)Detectron2是一个高效的开源目标检测框架，然而，由于其只聚焦目标检测和分割领域，该框架很难被仅仅做分类、回归、校准等其他任务的用户使用。因此，我们从Detectron2中剥离出框架的核心代码，取名为CustomKing，其追求仅仅提供框架，具体的任务场景由用户自己去加，方便更多的用户构建属于自己的科研或工程框架。

(2)安装CustomKing时不需要编译代码，而Detectron2需要编译。 因此，CustomKing可以轻松兼容Linux和Windows系统。

# 2. 使用CustomKing的优点是什么?
(1)CfgNode用于管理配置参数，方便实验和调试的参数设置。
