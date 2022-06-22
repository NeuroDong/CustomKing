from detectron2.data import DatasetCatalog
import os
import pandas as pd
import numpy as np

Classnumber=("0","1","2","3","4","5","6","7","8","9")

def load_process_data(dirnames,Classnumber):
    dicts=[]
    for dirname in dirnames:
        for label in Classnumber:
            label_csv_dir = os.path.join(dirname,label+".csv")
            data = pd.read_csv(label_csv_dir)
            data = np.array(data)
            for i in range(0,len(data),1):
                dicts.append({"image_name":data[i][0]+".jpg","x":data[i][2:],"y":label})
    return dicts


def register_process_data(name,dirname,Classnumber=Classnumber):
    DatasetCatalog.register(name, lambda: load_process_data(dirname, Classnumber))
    # # later, to access the data:
    data = DatasetCatalog.get(name)
    #print(data)