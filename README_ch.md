[English](https://github.com/dongdongdong1217/Detectron2-All/blob/main/README.md) | 简体中文

# 1. CustonKing是什么?
CustomKing是一个机器学习的框架，是用[Fvcore](https://github.com/facebookresearch/fvcore)写的一个类似于[Detectron2](https://github.com/facebookresearch/detectron2)的框架。 CustomKing与Detectron2的区别如下：

**1)** Detectron2是一个高效的开源目标检测框架，然而，由于其只聚焦目标检测和分割领域，该框架很难被做分类、回归、校准等其他任务的用户使用。因此，我们从Detectron2中剥离出框架的核心代码，取名为CustomKing，其目的是仅仅提供框架，具体的任务场景由用户自己去加，方便更多的用户构建属于自己的科研或工程框架。

**2)** 安装CustomKing时不需要编译代码，而Detectron2需要编译。 因此，CustomKing可以轻松兼容Linux和Windows系统。

# 2. 使用CustomKing的优点是什么?
**1)** CfgNode用于管理配置参数，方便实验和调试的参数设置。

**2)** Registry用于将算法模型变成一个可插拔的组件，方便通过名称直接调用网络，方便用户管理和使用大量的算法模型。

**3)** DatasetCatalog用于将数据集变成一个可插拔的组件，方便直接按名称调用数据集，方便用户管理和使用大量的数据集。

**4)** 如果使用者长期使用这个框架，可以一劳永逸地使用自己写入的模型、数据集、损失函数、优化器、Lr_schedule等。

# 3. 安装
```bash
pip intall -r requirements.txt
```

# 4.教程
请见[doc/CustomKing_User_Guide.pdf](https://github.com/NeuroDong/CustomKing/blob/main/doc/CustomKing_User_Guide.pdf)

# 5.示例
CustomKing里面内置了一个分类任务作为例子，包含了Resnet不同系列的模型，见customKing/modeling/meta_arch/Image_classification/Resnet.py，和完整的CIFAR-10数据集，见customKing/data/datasets/Image_classification/Cifar10.py。所有配置参数见customKing/config/defaults.py。采用如下代码运行实例即可：
```python
python tools/Image_Classification/main.py
```
