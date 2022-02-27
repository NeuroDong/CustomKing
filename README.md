# What is Detectron2-FC
We have been working hard in two directions: 

1)Quickly build complex deep learning network algorithm for training and prediction; 

2)It can quickly build deep learning algorithms for almost all tasks. 

Fortunately, detectron2 points out the direction for us. It builds a very convenient algorithm building platform based on pytorch, which can quickly reproduce the latest algorithms. However, detectron2 can only build algorithms with few tasks such as target detection and segmentation. For example, simple image classification cannot find convenience on detectron2. Therefore, on the basis of detectron2, we continue to enrich the functions of detectron2 to make it suitable for image classification, prediction, meta learning and other tasks.
# Installation tutorial
See https://detectron2.readthedocs.io/en/latest/tutorials/install.html for the installation of detectron2. The construction environment is as follows:

Ubuntu20.04

CUDA10.1

Pytorch1.8.1

# Using tutorials
All the built-in algorithms, datasets, and pre-trained weights can be trained or evaluated by running the following code, which just need change the parameters accordingly. For the selection of parameters, see 
https://github.com/dongdongdong1217/SENeXt-Transformer/tree/main/tools#readme.
```java  
  python3 tool/mian.py
```

# Existing examples
## A novel detection method for the close time of the taphole of blast furnace based on two-stage multimodal data fusion
See: https://github.com/dongdongdong1217/SENeXt-Transformer

## Object detection and image segmentation
Detectron2's own algorithm see: https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md

