from detectron2.data import DatasetCatalog
import os
import pandas as pd
import numpy as np

Classnumber=("0","1","2","3","4","5","6","7","8","9")

def load_process_and_icronWater(dirnames,Classnumber):

    dicts=[]
    process_dirname = dirnames[0]
    for label in Classnumber:
        label_csv_dir = os.path.join(process_dirname,label+".csv")
        data = pd.read_csv(label_csv_dir,header=None)
        process_data = np.array(data)

        image_label_dir = os.path.join(dirnames[1],label)
        image_path_list = os.listdir(image_label_dir)
        for image_name in image_path_list:
            image_path = os.path.join(image_label_dir,image_name)
            for i in range(0,len(process_data),1):
                if image_name[:-4] == process_data[i][0]:
                    dicts.append({"image_path":image_path,"x":process_data[i][2:],"y":label})
    return dicts

def register_process_and_icronWater(name,dirname,Classnumber=Classnumber):
    DatasetCatalog.register(name, lambda: load_process_and_icronWater(dirname, Classnumber))
    # # later, to access the data:
    data = DatasetCatalog.get(name)
    #print(data)
