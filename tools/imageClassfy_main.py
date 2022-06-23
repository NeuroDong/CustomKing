
from doctest import OutputChecker
from typing import Dict, List
from numpy.core.fromnumeric import mean
from torch import Tensor
from detectron2.data import get_detection_dataset_dicts
from detectron2.data import build_detection_train_loader,build_detection_test_loader
from detectron2.data.dataset_mapper import Cifar10 as Transforms #如果换数据集，这个数据预处理函数得换
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.checkpoint import DetectionCheckpointer,PeriodicCheckpointer
from sklearn.metrics import f1_score,average_precision_score
import torch.nn as nn

import numpy as np
import torch
from tqdm import tqdm
import os
import csv
import time
import logging
import logging.config


import detectron2.utils.comm as comm
from detectron2.engine import default_writers

from detectron2.utils.events import EventStorage

torch.backends.cudnn.benchmark = True
scaler = torch.cuda.amp.GradScaler()

def do_test(cfg, model):
        #导入数据
        dataset = get_detection_dataset_dicts(cfg.DATASETS.TEST)
        test_data = build_detection_test_loader(dataset,mapper=Transforms)
        model.eval()
        result_list = []
        label_list = []
        for data in tqdm(test_data):
            _,inference_result = model(data).cpu().detach().numpy()
            #print(inference_result)
            result = np.where(inference_result[0]==max(inference_result[0]))
            #print(result)
            result_list.append(result[0][0])
            label_list.append(int(float(data[0]["y"])))
        
        #把预测结果与标签保存到excel文件
        with open("./result_VS_label.csv", "a+", newline='', encoding='utf-8') as file:
            writer = csv.writer(file ,delimiter=',')
            writer.writerow(result_list)
        with open("./result_VS_label.csv", "a+", newline='', encoding='utf-8') as file:
            writer = csv.writer(file ,delimiter=',')
            writer.writerow(label_list)
        
        #计算accuracy,micro_f1_score和macron_f1_score
        correct = 0
        for i in range(0,len(label_list),1):
                if result_list[i] == label_list[i]:
                    correct = correct + 1
        accuracy = correct / len(label_list)
        print("accuracy:",accuracy)
        micro_f1_score = f1_score(label_list, result_list, average='micro')
        macro_f1_score = f1_score(label_list,result_list,average="macro")
        print("micro_f1_score:",micro_f1_score)
        print("macron_f1_score:",macro_f1_score)

        #计算AP和mAP
        class_dict = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
        for k in range(0,len(label_list),1):
                class_dict[label_list[k]].append(result_list[k])
        for t in range(0,10,1):
                with open("./class_dict2.csv", "a+", newline='', encoding='utf-8') as file:
                    writer = csv.writer(file ,delimiter=',')
                    writer.writerow(class_dict[t])
        ap_list = []
        for i in range(0,10,1):
                ap_label = []
                ap_result = []
                for j in range(0,len(label_list),1):
                    if label_list[j] == i:
                            ap_label.append(1)
                    else:
                            ap_label.append(0)
                    if result_list[j] == i :
                            ap_result.append(1)
                    else:
                            ap_result.append(0)
                AP = average_precision_score(ap_label,ap_result)
                ap_list.append(AP)
        mAP = mean(ap_list)
        print("ap_list:",ap_list)
        print("mAP:",mAP)

def do_train(cfg, model, resume=False):
        logging.basicConfig(level=logging.INFO) 
        model.train()
        optimizer = build_optimizer(cfg, model) 
        scheduler = build_lr_scheduler(cfg, optimizer) 
        checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler) 
        start_iter = (checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1) 
        dataset_train = get_detection_dataset_dicts(cfg.DATASETS.TRAIN)
        train_data = build_detection_train_loader(dataset_train,mapper=Transforms,total_batch_size=cfg.IMS_PER_BATCH,num_workers=0)
        max_iter = cfg.SOLVER.MAX_ITER
        cfg.SOLVER.CHECKPOINT_PERIOD = len(dataset_train) // cfg.IMS_PER_BATCH
        periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)
        writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

        dataset_test = get_detection_dataset_dicts(cfg.DATASETS.TEST)
        test_data = build_detection_test_loader(dataset_test,mapper=Transforms)

        with EventStorage(start_iter) as storage:
                time1 = time.time()
                correct = 0.0
                total = 0.0
                best_test_acc = 0.0
                best_test_iter = 0
                for data, iteration in zip(train_data, range(start_iter, max_iter)):
                        storage.iter = iteration
                        optimizer.zero_grad(set_to_none=True)

                        #--------------------数据预处理--------------------#
                        batchsize = len(data)
                        batch_images = []
                        batch_label = []
                        for i in range(0,batchsize,1):
                                dataValue_list = list(data[i].values())
                                batch_images.append(dataValue_list[0])
                                batch_label.append(dataValue_list[1])
                        batch_images_tensor = torch.stack(batch_images,dim=0).cuda().clone().detach().float()

                        with torch.cuda.amp.autocast():
                                predict = model(batch_images_tensor)

                        #--------------------计算损失值--------------------#
                        batch_label_tensor = torch.tensor(batch_label).cuda().float()
                        loss_fun = nn.CrossEntropyLoss()
                        loss = loss_fun(predict,batch_label_tensor.long())

                        time2 = time.time()
                        #---------------------更新权值---------------------#
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        #-------------计算训练准确率(即一个batch内的准确率)-----#
                        batch_label = []
                        for i in range(0,len(data),1):
                                batch_label.append(int(float(data[i]["y"])))
                        batch_label = torch.Tensor(batch_label).cuda()
                        _, predicted = torch.max(predict.data, 1)
                        
                        total += len(data)
                        correct += predicted.eq(batch_label).cpu().sum()
                        train_acc = correct / total

                        #---------------------记录并更新学习率--------------#
                        storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                        scheduler.step()
                        storage.put_scalar("loss", loss, smoothing_hint=False)
                        if iteration - start_iter > 5 and ((iteration + 1) % 20 == 0 or iteration == max_iter - 1):
                                storage.put_scalar("train_time",time.time()-time1,smoothing_hint=False)
                                storage.put_scalar("a_iter_backward_time",time.time()-time2,smoothing_hint=False)
                                storage.put_scalar("train_acc",train_acc,smoothing_hint=False)
                                time1 = time.time()
                                if iteration > 20 and iteration % cfg.SOLVER.CHECKPOINT_PERIOD <20:
                                        result_list = []
                                        label_list = []
                                        model.eval()
                                        with torch.no_grad():
                                                for data in tqdm(test_data,ncols=100):
                                                        #--------------------数据预处理--------------------#
                                                        batchsize = len(data)
                                                        batch_images = []
                                                        batch_label = []
                                                        for i in range(0,batchsize,1):
                                                                dataValue_list = list(data[i].values())
                                                                batch_images.append(dataValue_list[0])
                                                                batch_label.append(dataValue_list[1])
                                                        batch_images_tensor = torch.stack(batch_images,dim=0).cuda().clone().detach()

                                                        inference_result = model(batch_images_tensor).cpu().detach().numpy()
                                                        result = np.where(inference_result[0]==max(inference_result[0]))
                                                        result_list.append(result[0][0])
                                                        label_list.append(int(float(data[0]["y"])))
                                        correct_test = 0
                                        for i in range(0,len(label_list),1):
                                                if result_list[i] == label_list[i]:
                                                        correct_test = correct_test + 1
                                        test_acc = correct_test / len(label_list)
                                        if test_acc > best_test_acc:
                                                best_test_acc = test_acc
                                                best_test_iter = iteration
                                        storage.put_scalar("test_acc",test_acc,smoothing_hint=False)
                                else:
                                        storage.put_scalar("test_acc",np.NaN,smoothing_hint=False) 
                                for writer in writers:
                                        writer.write()
                        periodic_checkpointer.step(iteration)
                        model.train()
                print("Best test accuracy:",best_test_acc)
                print("Best test iteration:",best_test_iter)
                        

def main():
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        #--------------------设置配置参数-----------------------------#
        cfg = get_cfg() 
        cfg.MODEL.META_ARCHITECTURE = "Vit_small" #网络模型
        #------网络模型初始化需要传入的实参,有几个实参，就添加几个------#
        cfg.num_classes = 10
        cfg.ImageSize = 32
        cfg.DATASETS.TRAIN = "Cifar10_train" #训练数据集
        cfg.DATASETS.TEST = "Cifar10_test" #测试数据集
        cfg.JUST_EVAL = False #是否只是评估 
        cfg.PRE_WEIGHT = False
         #是否加载与训练权重
        cfg.IMS_PER_BATCH = 128 #batchsize
        #cfg.SOLVER.MAX_ITER = 15000 #训练最大iters
        cfg.OUTPUT_DIR = "output/Cifar10/ceshi_jit"
        cfg.CUDNN_BENCHMARK = True
        #-------------------------建立网络模型------------------------------#
        model = build_model(cfg)
        print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
        model = torch.jit.script(model) #如果报错且不知道怎么改，把这一行注释掉就行了
        #---------------------训练与测试------------------------------------#
        if cfg.JUST_EVAL:
                DetectionCheckpointer(model).load("pre_weights/Resnet_240.pth")#加载权值
                do_test(cfg,model)
        else:
                if cfg.PRE_WEIGHT:
                        DetectionCheckpointer(model).load("output/Cifar10/Resnet18_100000/model_0099999.pth")#加载权值
                do_train(cfg, model)
                #do_test(cfg,model) 


if __name__ == "__main__":
        main()
