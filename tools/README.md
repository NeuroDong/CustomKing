
This directory contains a few example scripts that demonstrate features of detectron2-FC,which does not include the original detectron2 (Original detectron2 see https://github.com/facebookresearch/detectron2/tree/main/tools ).

## Icron-water image classification
### First phase
Icron_water.py is a molten iron image classification task based on SE-Resnext101, run icron_water.py can train and test molten iron dataset. The loss function visualization program is shown in https://github.com/dongdongdong1217/Detectron2-FC/blob/main/visualizate/Classification/plt_smooth_trainloss.py. The convergence of the loss function in the training process is as follows:

![Image text](https://github.com/dongdongdong1217/Detectron2-FC/blob/main/visualizate/Visualization-diagram/Classification/Icron-water_trainloss.png)

| Model(SE-Resnext) | Backbone | train_dataset| test_dataset | Accuracy  | macron_f1_score | mAP |
| :----: |  :----: | :----: | :----: | :----: |:----: |:----: |
| 30000iters(batchsize=32) | Resnext101 | IcronWater_trainval2018+2021 |  IcronWater_test2018+2021 | 91.71%  |  91.72% | 87.62% |
| 40000iters(batchsize=32) | Resnext101 | IcronWater_trainval2018+2021 |  IcronWater_test2018+2021 | 92.48%  |  92.50% | 89.03% |
| 50000iters(batchsize=32) | Resnext101 | IcronWater_trainval2018+2021 |  IcronWater_test2018+2021 | 92.50%  |  92.51% | 88.87% |
| 30000iters(batchsize=32) | Resnext101 | IcronWater_trainval2021 |  IcronWater_test2021 | 95.71%  |  95.71% | 93.64% |
### Second phase
The convergence of the loss function in the training process is as follows:

![Image text](https://github.com/dongdongdong1217/Detectron2-FC/blob/main/visualizate/Visualization-diagram/Classification/Icron-water_xifen_trainloss.png)

| Model(SE-Resnext) | Backbone | train_dataset| test_dataset | Accuracy  | macron_f1_score | mAP |
| :----: |  :----: | :----: | :----: | :----: |:----: |:----: |
| 5000iters(batchsize=32) | Resnext101 | IcronWater2021_train_xifen |  IcronWater2021_train_xifen |  61.83% |  62.16% | 46.25% |
| 10000iters(batchsize=32) | Resnext101 | IcronWater2021_train_xifen |  IcronWater2021_train_xifen |  64.23% |  63.77% | 46.77% |
| 15000iters(batchsize=32) | Resnext101 | IcronWater2021_train_xifen |  IcronWater2021_train_xifen |  63.94% |  63.35% | 46.18% |

## Process-data classification
### First phase
process_data.py  is a blast furnace ironmaking process data classification task based on Transformer, run process_data.py can train and test the blast furnace ironmaking process data. The loss function visualization program is shown in https://github.com/dongdongdong1217/Detectron2-FC/blob/main/visualizate/Classification/plt_smooth_trainloss.py. The convergence of the loss function in the training process is as follows:

![Image text](https://github.com/dongdongdong1217/Detectron2-FC/blob/main/visualizate/Visualization-diagram/Classification/Process_data_trainloss.png)

| Model(SE-Resnext) | Backbone | train_dataset| test_dataset | Accuracy  | macron_f1_score | mAP |
| :----: |  :----: | :----: | :----: | :----: |:----: |:----: |
| 999iters(batchsize=32) | Transformer | Process_data2021_trainval |  Process_data2021_test | 89.81%  |  86.04% | 60.90% |
| 1999iters(batchsize=32) | Transformer | Process_data2021_trainval |  Process_data2021_test | 89.81%  |  86.09% | 60.69% |
| 5000iters(batchsize=32) | Transformer |  Process_data2021_trainval |  Process_data2021_test | 89.81%  |  86.59% | 70.49% |

### Second phase
The convergence of the loss function in the training process is as follows:
![Image text](https://github.com/dongdongdong1217/Detectron2-FC/blob/main/visualizate/Visualization-diagram/Classification/Process_data_xifen_trainloss.png)
| Model(SE-Resnext) | Backbone | train_dataset| test_dataset | Accuracy  | macron_f1_score | mAP |
| :----: |  :----: | :----: | :----: | :----: |:----: |:----: |
| 1999iters(batchsize=32) | Transformer | Process_data2021_train_xifen |  Process_data2021_test_xifen | 86.43%  |  85.74% | 51.80% |
| 2999iters(batchsize=32) | Transformer | Process_data2021_train_xifen |  Process_data2021_test_xifen | 86.43%  |  85.74% | 51.80% |

AP values for each category are as follows：
| Category 1 | Category 2 | Category 3| Category 4 | Category 5  | Category 6 | Category 7 | Category 8 | Category 9 | Category 10 |
| :----: |  :----: | :----: | :----: | :----: |:----: |:----: |:----: |:----: |:----: |
| 13.57% | 22.95% | 33.45% | 47.19% | 55.67% | 61.17% | 65.99% | 69.19% | 72.79% | 76.06% |


