from unicodedata import name
from detectron2.data import DatasetCatalog
import os
import numpy as np
import pickle
import torchvision
from torchvision import transforms
import torch.utils.data as torchdata

Classnumber=("0","1","2","3","4","5","6","7","8","9")

def load_Cifar10(name,root):
    # dicts=[]
    # for dirname in dirnames:
    #     for label in Classnumber:
    #         label_dir = os.path.join(dirname,label)
    #         image_path_list = os.listdir(label_dir)
    #         for image_name in image_path_list:
    #             image_path = os.path.join(label_dir,image_name)
    #             dicts.append({"file_name":image_path,"label":label})
    # return dicts

    if name == "Cifar10_train":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
            transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
            ])
        dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train) #训练数据集
    if name =="Cifar10_test":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    if name == "Cifar10_train_and_test":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
            transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
            ])
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train) #训练数据集
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
        dataset = torchdata.ConcatDataset([trainset,testset])
    return dataset
    

def register_Cifar10(name,root):
    DatasetCatalog.register(name, lambda: load_Cifar10(name,root))
    #data = DatasetCatalog.get(name)



