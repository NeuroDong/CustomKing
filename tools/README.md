
This directory contains a few example scripts that demonstrate features of detectron2-FC,which does not include the original detectron2 (See https://github.com/facebookresearch/detectron2/tree/main/tools ).

## Icron-water image classification
Icron_water.py is a Icon-water image classification task based on SE-Resnext101, run icron_water.py can train and test Icron-water dataset.The loss function visualization program is shown in https://github.com/dongdongdong1217/Detectron2-FC/blob/main/visualizate/Classification/plt_smooth_trainloss.py. The convergence of the loss function in the training process is as follows:

![Image text](https://github.com/dongdongdong1217/Detectron2-FC/blob/main/visualizate/Visualization-diagram/Classification/Icron-water_trainloss.png)

| Model(SE-Resnext) | Backbone | train_dataset| test_dataset | Accuracy  | macron_f1_score | mAP |
| :----: |  :----: | :----: | :----: | :----: |:----: |:----: |
| 30000iters(batchsize=32) | Resnext101 | IcronWater_trainval2018+2021 |  IcronWater_test2018+2021 | 91.71%  |  91.72% | 87.62% |
| 40000iters(batchsize=32) | Resnext101 | IcronWater_trainval2018+2021 |  IcronWater_test2018+2021 | 92.48%  |  92.50% | 89.03% |
| 50000iters(batchsize=32) | Resnext101 | IcronWater_trainval2018+2021 |  IcronWater_test2018+2021 | 92.50%  |  92.51% | 88.87% |
| 30000iters(batchsize=32) | Resnext101 | IcronWater_trainval2021 |  IcronWater_test2021 | 95.71%  |  95.71% | 93.64% |

## Process-data classification
process_data.py  is a blast furnace ironmaking process data classification task based on Transformer, run process_data.py can train and test the blast furnace ironmaking process data. The loss function visualization program is shown in https://github.com/dongdongdong1217/Detectron2-FC/blob/main/visualizate/Classification/plt_smooth_trainloss.py. The convergence of the loss function in the training process is as follows:

![Image text](https://github.com/dongdongdong1217/Detectron2-FC/blob/main/visualizate/Visualization-diagram/Classification/Process_data_trainloss.png)

| Model(SE-Resnext) | Backbone | train_dataset| test_dataset | Accuracy  | macron_f1_score | mAP |
| :----: |  :----: | :----: | :----: | :----: |:----: |:----: |
| 5000iters(batchsize=32) | Transformer | Process_data2021 |  Process_data2021 | 89.81%  |  86.59% | 70.49% |
