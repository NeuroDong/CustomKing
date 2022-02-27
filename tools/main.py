
from asyncio import FastChildWatcher
from numpy.core.fromnumeric import mean
from detectron2.data import get_detection_dataset_dicts
from detectron2.data import build_detection_train_loader,build_detection_test_loader
from detectron2.data.dataset_mapper import process_and_icronWater
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.checkpoint import DetectionCheckpointer,PeriodicCheckpointer
from sklearn.metrics import f1_score,average_precision_score

import numpy as np
import torch
from tqdm import tqdm
import os



import detectron2.utils.comm as comm
from detectron2.engine import default_writers

from detectron2.utils.events import EventStorage

def do_test(cfg, model):
      #导入数据
      dataset = get_detection_dataset_dicts(cfg.DATASETS.TEST)
      test_data = build_detection_test_loader(dataset,mapper=process_and_icronWater)
      model.eval()
      result_list = []
      label_list = []
      for data in tqdm(test_data):
            inference_result = model(data).cpu().detach().numpy()
            result = np.where(inference_result[0]==max(inference_result[0]))
            result_list.append(result[0][0])
            label_list.append(int(float(data[0]["y"])))
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
      train_data = build_detection_train_loader(dataset,mapper=process_and_icronWater,total_batch_size=cfg.IMS_PER_BATCH)

      with EventStorage(start_iter) as storage:
            for data, iteration in zip(train_data, range(start_iter, max_iter)):
                  storage.iter = iteration
                  loss = model(data)
                  
                  #---------------------更新权值---------------------#
                  optimizer.zero_grad()
                  loss.backward()
                  optimizer.step()
                  #---------------------记录并更新学习率--------------#
                  storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                  scheduler.step()

                  storage.put_scalar("loss", loss, smoothing_hint=False)
                  if iteration - start_iter > 5 and ((iteration + 1) % 20 == 0 or iteration == max_iter - 1):
                        for writer in writers:
                              writer.write()
                        print("iters:{},loss:{}".format(iteration,loss))
                              
                  periodic_checkpointer.step(iteration)


def main():
      os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
      #--------------------设置配置参数-----------------------------#
      cfg = get_cfg() 
      cfg.MODEL.META_ARCHITECTURE = "Resnet_encoder" #网络模型
      cfg.src_vocab_size = 1001 #输入字典的长度
      cfg.tgt_vocab_size = 10 #输出字典的长度
      cfg.DATASETS.TRAIN = "process_and_icronWater_train" #训练数据集
      cfg.DATASETS.TEST = "process_and_icronWater_test" #测试数据集
      cfg.JUST_EVAL = False #是否只是评估
      cfg.PRE_WEIGHT = False #是否加载与训练权重
      cfg.IMS_PER_BATCH = 4 #batchsize
      cfg.SOLVER.MAX_ITER = 50000 #训练最大iters
      cfg.SOLVER.CHECKPOINT_PERIOD = 1000 #每个多少iters保存一次权值
      cfg.OUTPUT_DIR = "output/classification/Resnet_encoder"
      cfg.CUDNN_BENCHMARK = True
      print(cfg)


      #-------------------------建立网络模型------------------------------#
      model = build_model(cfg)
      print(model)

      #---------------------训练与测试------------------------------------#
      if cfg.JUST_EVAL:
            DetectionCheckpointer(model).load("output/classification/Resnet_encoder/model_0004999.pth")#加载权值
            do_test(cfg,model)
      else:
            if cfg.PRE_WEIGHT:
                  DetectionCheckpointer(model).load("output/classification/process_and_icronWater_enc49/model_0014999.pth")#加载权值
            do_train(cfg, model)
            do_test(cfg,model) 

if __name__ == "__main__":
      main()