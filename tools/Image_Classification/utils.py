import torch
import logging
from customKing.modeling.meta_arch.build import build_model
from customKing.solver.build import build_optimizer,build_lr_scheduler
from customKing.data import get_dataset_dicts,build_loader 
from customKing.engine import default_writers
import customKing.utils.comm as comm
from customKing.utils.events import EventStorage
import time
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import json
from torch.nn import DataParallel
import os

class doTrain():
    def __init__(self,cfg):
        self.iteration = cfg.SOLVER.START_ITER
        self.cfg = cfg

        self.dataset_train = get_dataset_dicts(self.cfg.DATASETS.TRAIN)    
        self.train_data = build_loader(self.cfg,self.dataset_train)    
        if self.cfg.DATASETS.VALID != "":
            self.dataset_valid = get_dataset_dicts(self.cfg.DATASETS.VALID)    
            self.valid_data = build_loader(self.cfg,self.dataset_valid)   
        self.testClass = doTest(cfg)
        self.best_model = None

    def Inference_Output(self,Dataloader,data_mode="Valdata"):
        if self.best_model == None:
            self.load_best_model()
        Data_model = self.best_model
        Data_model.eval()
        if "/" in self.cfg.MODEL.OUTPUT_DIR:
            filename = self.cfg.MODEL.OUTPUT_DIR.split("/")
        elif "\\" in self.cfg.MODEL.OUTPUT_DIR:
            filename = self.cfg.MODEL.OUTPUT_DIR.split("\\")

        with torch.no_grad():
            jishu = 0.
            for batch_img,batch_label in Dataloader:
                batch_img = batch_img.cuda().clone().detach().float()
                predict = Data_model(batch_img,batch_label).to(torch.float64)
                for i in range(0,batch_img.shape[0],1):
                    num = jishu//30000
                    with open(self.cfg.MODEL.OUTPUT_DIR[:-(len(filename[-1]))]+"/"+data_mode+f"_before_calibration{int(num)}.json","a+",encoding="UTF-8") as fp:
                            fp.write(json.dumps([predict[i].tolist(),batch_label[i].item()])+"\n")
                    jishu = jishu +1

        print(data_mode+"Calibration data (before calibration) written successfully!")

    def load_best_model(self):
        best_model_folder = self.cfg.MODEL.OUTPUT_DIR[:-12]
        best_model_path = None
        for file in os.listdir(best_model_folder):
            if file[-4:] == ".pth":
                best_model_path = os.path.join(best_model_folder,file)
        assert best_model_path != None,"Could not find the path to the best model from "+ best_model_folder
        self.best_model = build_model(self.cfg)
        if self.cfg.SOLVER.IS_PARALLEL:
            pretrained_dict = torch.load(best_model_path).module.state_dict() 
            model_dict = self.best_model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            self.best_model.load_state_dict(pretrained_dict)
        else:
                self.best_model = torch.load(best_model_path)

    def do_train(self, model):
        logging.basicConfig(level=logging.INFO) 
        model.train() 
        optimizer = build_optimizer(self.cfg, model) 
        scheduler = build_lr_scheduler(self.cfg, optimizer) 
        if self.cfg.SOLVER.IS_PARALLEL:
            model = DataParallel(model,device_ids=[i for i in range(torch.cuda.device_count())])

        writers = default_writers(self.cfg.MODEL.OUTPUT_DIR, self.cfg.SOLVER.MAX_ITER) if comm.is_main_process() else []          

        with EventStorage(self.cfg.SOLVER.START_ITER) as storage:
            for epoch in range(self.cfg.SOLVER.MAX_EPOCH):
                time1 = time.time()
                for batch_img,batch_label in self.train_data:
                    self.iteration = self.iteration + 1
                    if self.iteration > self.cfg.SOLVER.MAX_ITER:
                        break
                    storage.iter = self.iteration 
                    batch_img = batch_img.cuda().clone().detach().float()    
                    batch_label = batch_label.cuda().float().long()                        
                    predict,losses = model(batch_img,batch_label)  
                    time2 = time.time()
                    optimizer.zero_grad()   
                    if self.cfg.SOLVER.IS_PARALLEL:
                        loss = losses.sum()/len(losses)
                        loss.backward()
                    else:
                        loss = losses
                        loss.backward()   
                    optimizer.step()  

                    #---------Calculate the training accuracy (that is, the accuracy within a batch)-----#
                    _, predicted = torch.max(predict.data, 1)
                    correct = predicted.eq(batch_label).cpu().sum()
                    train_acc = correct / batch_img.shape[0]

                    #--------------Record and update the learning rate--------------#
                    storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                    scheduler.step()
                    storage.put_scalar("loss", loss, smoothing_hint=False)

                    if self.iteration - self.cfg.SOLVER.START_ITER > 5 and ((self.iteration + 1) % 20 == 0 or self.iteration == self.cfg.SOLVER.MAX_ITER - 1):
                            storage.put_scalar("train_time",time.time()-time1,smoothing_hint=False)
                            storage.put_scalar("a_iter_backward_time",time.time()-time2,smoothing_hint=False)
                            storage.put_scalar("train_acc",train_acc,smoothing_hint=False)
                            time1 = time.time()
                            for writer in writers:
                                writer.write()

                self.testClass.do_test(model,self.iteration,storage)  #Evaluate on the test set every other epoch
                model.train()
                if self.iteration > self.cfg.SOLVER.MAX_ITER:
                    break
                
        #Record inference results on the training set (for model calibration)
        self.Inference_Output(self.train_data,data_mode="Traindata")

        #Record inference results on the training set (for model calibration)
        self.Inference_Output(self.valid_data,data_mode="Validdata")

        #Record inference results on the training set (for model calibration)
        self.Inference_Output(self.testClass.test_data,data_mode="Testdata")

        print("Best test accuracy:",self.testClass.best_test_acc)
        print("Best test iteration:",self.testClass.best_test_iter)

class doTest():
    def __init__(self,cfg) -> None:
        self.cfg = cfg
        self.dataset_test = get_dataset_dicts(self.cfg.DATASETS.TEST)  
        self.test_data = build_loader(self.cfg,self.dataset_test)  
        self.best_test_acc = 0.
        self.best_test_iter = 0

    def do_test(self,model,iteration=None,storage=None):
        result_list = []
        label_list = []
        model.eval()
        with torch.no_grad():
            for batch_img,batch_label in tqdm(self.test_data,dynamic_ncols=True):
                batch_img = batch_img.cuda().clone().detach().float()  
                inference_result = model(batch_img,batch_label)
                _, result = torch.max(inference_result.data, 1)
                result_list = result_list + result.tolist()
                label_list = label_list + batch_label.tolist()
        correct = 0
        for i in range(len(label_list)):
            if label_list[i] == result_list[i]:
                correct = correct + 1
        test_acc = correct / len(label_list)
        if iteration != None:
            if test_acc > self.best_test_acc:
                    self.best_test_acc = test_acc
                    self.best_test_iter = iteration
                    if "/" in self.cfg.MODEL.OUTPUT_DIR:
                        filename = self.cfg.MODEL.OUTPUT_DIR.split("/")
                    elif "\\" in self.cfg.MODEL.OUTPUT_DIR:
                        filename = self.cfg.MODEL.OUTPUT_DIR.split("\\")
                    for file in os.listdir(self.cfg.MODEL.OUTPUT_DIR[:-(len(filename[-1]))]):
                        if file[-4:] == ".pth":
                            os.remove(os.path.join(self.cfg.MODEL.OUTPUT_DIR[:-(len(filename[-1]))],file))
                    torch.save(model, self.cfg.MODEL.OUTPUT_DIR[:-(len(filename[-1]))] +str(float(self.best_test_iter))[:-2]+".pth")
        else:
            print("ACC of test data:",test_acc)
        if storage != None:
            storage.put_scalar("test_acc",test_acc,smoothing_hint=False)
