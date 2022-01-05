from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader,build_detection_test_loader
from detectron2.data import get_detection_dataset_dicts
from detectron2.data.dataset_mapper import process_data_mapper

cfg = get_cfg() 
cfg.DATASETS.TRAIN = "process_data_trainval" #训练数据集
dataset = get_detection_dataset_dicts(cfg.DATASETS.TRAIN)
train_data = build_detection_train_loader(dataset,mapper=process_data_mapper,total_batch_size=2)
for data in train_data:
    print(data)
