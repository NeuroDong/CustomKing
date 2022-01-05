
This directory contains a few example scripts that demonstrate features of detectron2-FC,which does not include the original detectron2 (See https://github.com/facebookresearch/detectron2/tree/main/tools ).

## Icron-water image classification
Icron_water.py is a Icon-water image classification task based on SE-Resnext101, Run icron_water.py can train and test Icron-water dataset.The loss function visualization program is shown in https://github.com/dongdongdong1217/Detectron2-FC/blob/main/visualizate/Classification/plt_smooth_trainloss.py. The convergence of the loss function in the training process is as follows:
![Image text](https://github.com/dongdongdong1217/Detectron2-FC/blob/main/visualizate/Visualization-diagram/Classification/Icron-water_trainloss.png)

| Model(SE-Resnext) | Backbone | train_dataset| test_dataset | Accuracy  | macron_f1_score | mAP |
| :----: |  :----: | :----: | :----: | :----: |:----: |:----: |
| 30000iters(batchsize=32) | Resnext101 | IcronWater_trainval2018+2021 |  IcronWater_test2018+2021 | 91.71%  |  91.72% | 87.62% |
| 40000iters(batchsize=32) | Resnext101 | IcronWater_trainval2018+2021 |  IcronWater_test2018+2021 | 92.48%  |  92.50% | 89.03% |
| 50000iters(batchsize=32) | Resnext101 | IcronWater_trainval2018+2021 |  IcronWater_test2018+2021 | 92.50%  |  92.51% | 88.87% |
| 30000iters(batchsize=32) | Resnext101 | IcronWater_trainval2021 |  IcronWater_test2021 | 95.71%  |  95.71% | 93.64% |
