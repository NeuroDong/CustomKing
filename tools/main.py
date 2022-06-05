
from numpy.core.fromnumeric import mean
from detectron2.data import get_detection_dataset_dicts
from detectron2.data import build_detection_train_loader,build_detection_test_loader
from detectron2.data.dataset_mapper import Cifar10 as Transforms #如果换数据集，这个数据预处理函数得换
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.checkpoint import DetectionCheckpointer,PeriodicCheckpointer
from sklearn.metrics import f1_score,average_precision_score

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
            inference_result = model(data).cpu().detach().numpy()
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
        max_iter = cfg.SOLVER.MAX_ITER
        periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)
        writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

        #---------------------------导入数据--------------------------------#
        dataset = get_detection_dataset_dicts(cfg.DATASETS.TRAIN)
        train_data = build_detection_train_loader(dataset,mapper=Transforms,total_batch_size=cfg.IMS_PER_BATCH,num_workers=4)

        with EventStorage(start_iter) as storage:
                time1 = time.time()
                for data, iteration in zip(train_data, range(start_iter, max_iter)):
                    storage.iter = iteration
                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast():
                        loss = model(data)
                    time2 = time.time()
                    #---------------------更新权值---------------------#
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    #loss.backward()
                    #optimizer.step()
                    #---------------------记录并更新学习率--------------#
                    storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                    scheduler.step()
                    storage.put_scalar("loss", loss, smoothing_hint=False)
                    if iteration - start_iter > 5 and ((iteration + 1) % 20 == 0 or iteration == max_iter - 1):
                            storage.put_scalar("train_time",time.time()-time1,smoothing_hint=False)
                            storage.put_scalar("a_iter_backward_time",time.time()-time2,smoothing_hint=False)
                            time1 = time.time()
                            for writer in writers:
                                writer.write()
                            
                    periodic_checkpointer.step(iteration)


def main():
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        #--------------------设置配置参数-----------------------------#
        cfg = get_cfg() 
        cfg.MODEL.META_ARCHITECTURE = "Complete_graph_network18" #网络模型
        #------网络模型初始化需要传入的实参,有几个实参，就添加几个------#
        cfg.num_classes = 10
        cfg.ImageSize = 32
        cfg.DATASETS.TRAIN = "Cifar10_train" #训练数据集
        cfg.DATASETS.TEST = "Cifar10_test" #测试数据集
        cfg.JUST_EVAL = False #是否只是评估 
        cfg.PRE_WEIGHT = False
         #是否加载与训练权重
        cfg.IMS_PER_BATCH = 128 #batchsize
        cfg.SOLVER.MAX_ITER = 64000 #训练最大iters
        cfg.SOLVER.CHECKPOINT_PERIOD = 1000 #每个多少iters保存一次权值
        cfg.OUTPUT_DIR = "output/Cifar10/Complete_graph_network18_data_enhancement_no_pool"
        cfg.CUDNN_BENCHMARK = True
        print(cfg)
        #-------------------------建立网络模型------------------------------#
        model = build_model(cfg)
        #print(model)
        print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
        #---------------------训练与测试------------------------------------#
        if cfg.JUST_EVAL:
                DetectionCheckpointer(model).load("output/Cifar10/Complete_graph_network18_data_enhancement/model_0063999.pth")#加载权值
                do_test(cfg,model)
        else:
                if cfg.PRE_WEIGHT:
                        DetectionCheckpointer(model).load("pre_weights/vit_b_16-c867db91.pth")#加载权值
                do_train(cfg, model)
                do_test(cfg,model) 


if __name__ == "__main__":
        main()