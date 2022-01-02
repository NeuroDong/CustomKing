
This directory contains a few example scripts that demonstrate features of detectron2-FC,which does not include the original detectron2 (See https://github.com/facebookresearch/detectron2/tree/main/tools ).

## Icron-water image classification
Icron_water.py is a Icon-water image classification task based on SE-Resnext101,dataset see:https://pan.baidu.com/s/157NWRH7Wf4YE_JgVCqmvew, 
extraction code is xoyi. 

Run icron_water.py can train and test Icron-water dataset.The loss function visualization program is shown in https://github.com/dongdongdong1217/Detectron2-FC/blob/main/visualizate/Classification/plt_smooth_trainloss.py. The convergence of the loss function in the training process is as follows:
![Image text](https://github.com/dongdongdong1217/Detectron2-FC/blob/main/visualizate/Visualization-diagram/Classification/Icron-water_trainloss.png)

| Model        | Backbone    | Accuracy  |
| --------   | -----:   | :----: |
| SE-Resnext        | Resnext101     |   92.50%    |
