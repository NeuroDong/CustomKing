English | [简体中文](https://github.com/dongdongdong1217/Detectron2-All/blob/main/README_ch.md)

![Image text](https://github.com/dongdongdong1217/Detectron2-All/blob/main/docs/NeuroDong3.jpg)

# What is Detectron2-All
Detectron2-All is a fast neural network construction platform based on Detectron2, and its difference with Detectron2 is that Detectron2 only has built-in algorithms and datasets in the fields of object detection, image segmentation, etc., and the field is relatively narrow, while Detectron2-All is committed to building all common machine learning algorithms (mainly deep learning) and all commonly used data sets, including but not limited to classification, regression, few-shot, and Meta-learning.
# Installation tutorial
See https://detectron2.readthedocs.io/en/latest/tutorials/install.html for the installation of detectron2. The construction environment is as follows:

Ubuntu20.04

CUDA10.1

Pytorch1.8.1

# Using tutorials
## Use the detectron2-All built-in algorithms and datasets
All the built-in algorithms, datasets, and pre-trained weights can be trained or evaluated by running the following code, which just need change the parameters accordingly. For the selection of parameters, see 
https://github.com/dongdongdong1217/Detectron2-All/tree/main/tools#readme.
```java  
  python3 tool/mian.py
```
## Use custom networks and data
See: https://detectron2.readthedocs.io/en/latest/tutorials/index.html, or contact the author for a discussion. The author's email address is dongjinzong@126.com, WeChat QR code see the logo image above.

# Existing examples
## A novel detection method for the close time of the taphole of blast furnace based on two-stage multimodal data fusion
See: https://github.com/dongdongdong1217/SENeXt-Transformer

## Object detection and image segmentation
Detectron2's own algorithm see: https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md

