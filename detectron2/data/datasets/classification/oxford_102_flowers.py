from detectron2.data import DatasetCatalog
import os
import numpy as np
import pickle

Classnumber=("0","1","2","3","4","5","6","7","8","9")

def load_flowers102(dirnames,Classnumber=Classnumber):
    dicts=[]
    for dirname in dirnames:
        for label in Classnumber:
            label_dir = os.path.join(dirname,label)
            image_path_list = os.listdir(label_dir)
            for image_name in image_path_list:
                image_path = os.path.join(label_dir,image_name)
                dicts.append({"file_name":image_path,"label":label})
    return dicts

def register_flowers102(name,dirname):
    DatasetCatalog.register(name, lambda: load_flowers102(dirname))
    data = DatasetCatalog.get(name)