Please change the parameter for algorithm selection: cfg.MODEL.META_ARCHITECTURE, which has the following options.

# Image Classification Network
```java  
se_resnext_101
VIT
SwinTransformer
ResNeXt101
Sk_coatnet_0
```
# Multimodal Data Fusion Network
```java
Se_resnext_tranformer
SENeXt_Transformer
SENeXt_Encoder
SENeXt_Decoder
Se_resnext_Decoder
Se_resnext_Encoder
Resnext_tranformer
Resnext_decoder
Resnext_encoder
Resnet_tranformer
Resnet_decoder
Resnet_encoder
```
# Multivariable Classification
```java
Transformer_cls
```


Please change the parameter for training data set selection: cfg.DATASETS.TRAIN, which has the following options.
```java 
icron_water_trainval
icron_water_train
icron_water_val
IcronWater2021_trainval
IcronWater2021_train
IcronWater2021_train_xifen
process_data_trainval
process_data_train
process_data_train_xifen
process_and_icronWater_train：The fusion data set of the second stage.
```

Please change the parameter for test data set selection: cfg.DATASETS.TEST, which has the following options.
```java 
icron_water_test
IcronWater2021_test
IcronWater2021_test_xifen
process_data_test
process_data_test_xifen
process_and_icronWater_test
```
