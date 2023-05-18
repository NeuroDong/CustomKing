English | [简体中文](https://github.com/dongdongdong1217/Detectron2-All/blob/main/README_ch.md)

# 1. What is CustomKing
CustomKing is a framework for machine learning, which is a framework similar to Detectron2 written by Fvcore. The differences between CustomKing and Detectron2 are as follows:

**1)** Detectron2 is an efficient open-source object detection framework. However, because it only focuses on target detection and segmentation, the framework is difficult to be used by users who only do classification, regression, calibration, and other tasks. Therefore, we stripped the core code of the framework from Detectron2 and named it CustomKing. CustomKing's pursuit is only to provide the framework, and the specific task scenarios are added by the users of CustomKing so that more users can build their own scientific research or engineering frameworks.

**2)** The code does not need to be compiled when installing CustomKing, while Detectron2 needs to be compiled. Therefore, CustomKing can easily be compatible with the Linux and Windows systems.

# 2. Benefits of using CustomKing
**1)** CfgNode is used to manage configuration parameters, which is convenient for parameter setting of experiments and debugging.

**2)** The Registry is used to turn the algorithm model into a pluggable component, which is convenient for directly calling the network by name, and is convenient for users to manage and use a large number of algorithm models.

**3)** The DatasetCatalog is used to turn the dataset into a pluggable component, which is convenient for directly calling the dataset by name, and is convenient for users to manage and use a large number of datasets.

**4)** If users use this framework for a long time, they can use their written models, data sets, loss functions, optimizers, Lr_schedule, etc., once and for all.

# 3. Tutorial

# 4. Example
CustomKing has a built-in classification task as an example, including different series of Resnet models, see customKing/modeling/meta_arch/Image_classification/Resnet.py, and the complete CIFAR-10 dataset, see customKing/data/datasets/Image_classification/Cifar10 .py. See customKing/config/defaults.py for all configuration parameters. Run the example with the following code:

'''
python tools/Image_Classification/main.py
'''
