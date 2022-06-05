English | [简体中文](https://github.com/dongdongdong1217/Detectron2-All/blob/main/README_ch.md)

![Image text](https://github.com/dongdongdong1217/Detectron2-All/blob/main/docs/NeuroDong3.jpg)

# 1. What is Detectron2-All
Detectron2-All is a fast neural network construction platform based on Detectron2, and its difference with Detectron2 is that Detectron2 only has built-in algorithms and datasets in the fields of object detection, image segmentation, etc., and the field is relatively narrow, while Detectron2-All is committed to building all common machine learning algorithms (mainly deep learning) and all commonly used data sets, including but not limited to classification, regression, few-shot, and Meta-learning.
# 2. Installation tutorial
See https://detectron2.readthedocs.io/en/latest/tutorials/install.html for the installation of detectron2. The construction environment is as follows:

Ubuntu20.04

CUDA10.1

Pytorch1.8.1

# 3. Using tutorials
## 3.1 Use the detectron2-All built-in algorithms and datasets
All the built-in algorithms, datasets, and pre-trained weights can be trained or evaluated by running the following code, which just need change the parameters accordingly.
```java  
  python3 tool/mian.py
```
Algorithm selection method: in main.py file:
```java  
cfg.MODEL.META_ARCHITECTURE = "Configuration parameter name" 
```
See the table in section 4 for configuration parameter names.

Dataset selection method: in main.py file:
```java  
cfg.DATASETS.TRAIN = "Configuration parameter name" #train set
cfg.DATASETS.TEST = "Configuration parameter name" #test set
```
See the table in section 5 for configuration parameter names.

If you want to change more configuration parameters, see: https://github.com/dongdongdong1217/Detectron2-All/blob/main/detectron2/config/defaults.py.
## 3.2 Use custom networks and data
See: https://detectron2.readthedocs.io/en/latest/tutorials/index.html, or contact the author for a discussion. The author's email address is dongjinzong@126.com, WeChat QR code see the logo image above.

# 4. Existing built-in algorithms


