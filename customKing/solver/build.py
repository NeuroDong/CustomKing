import torch
from customKing.config.config import CfgNode
import math
from torch.nn.functional import softmax,relu
import numpy as np
import torch.nn as nn

__all__=["build_optimizer","build_lr_scheduler"]

# ---------------------------------------------------------------------------- #
# build optimizer
# ---------------------------------------------------------------------------- #
def build_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    
    if cfg.SOLVER.OPTIMIZER == "SGD":
        optimizer = _get_SGD_optimizer(cfg.SOLVER,model)
    if cfg.SOLVER.OPTIMIZER == "Adam":
        optimizer = _get_adam_optimizer(cfg.SOLVER,model)
    
    return optimizer

def build_Calibration_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    if cfg.CALIBRATION_SOLVER.OPTIMIZER == "SGD":
        optimizer = _get_SGD_optimizer(cfg.CALIBRATION_SOLVER,model)
    if cfg.CALIBRATION_SOLVER.OPTIMIZER == "Adam":
        optimizer = _get_adam_optimizer(cfg.CALIBRATION_SOLVER,model)
    return optimizer

def _get_SGD_optimizer(Solver,model:torch.nn.Module):
    return torch.optim.SGD(model.parameters(),lr=Solver.BASE_LR,momentum=Solver.MOMENTUM,weight_decay=Solver.WEIGHT_DECAY,nesterov=Solver.NESTEROV)

def _get_adam_optimizer(Solver,model:torch.nn.Module):
    return torch.optim.Adam(model.parameters(),lr=Solver.BASE_LR,weight_decay=Solver.WEIGHT_DECAY)


def build_Calibration_optimizer_for_all(cfg: CfgNode, model: torch.nn.Module,i) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    if cfg.CALIBRATION_SOLVER.OPTIMIZER == "SGD":
        optimizer = _get_SGD_optimizer_for_all(cfg.CALIBRATION_SOLVER,model,i)
    if cfg.CALIBRATION_SOLVER.OPTIMIZER == "Adam":
        optimizer = _get_adam_optimizer_for_all(cfg.CALIBRATION_SOLVER,model,i)
    return optimizer

def _get_SGD_optimizer_for_all(Solver,model:torch.nn.Module,i):
    return torch.optim.SGD(filter(lambda p : p.requires_grad, model.parameters()),lr=Solver.BASE_LRS[i],momentum=Solver.MOMENTUM,weight_decay=Solver.WEIGHT_DECAY,nesterov=Solver.NESTEROV)

def _get_adam_optimizer_for_all(Solver,model:torch.nn.Module,i):
    return torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()),lr=Solver.BASE_LRS[i],weight_decay=Solver.WEIGHT_DECAY)


# ---------------------------------------------------------------------------- #
# build lr_scheduler
# ---------------------------------------------------------------------------- #
def build_lr_scheduler(cfg: CfgNode, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build an scheduler from config.
    """
    Scheduler = Schedulers(cfg.SOLVER,optimizer)
    return Scheduler.Scheduler

def build_Calibration_lr_scheduler(cfg: CfgNode, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build an scheduler from config.
    """
    Scheduler = Schedulers(cfg.CALIBRATION_SOLVER,optimizer)
    return Scheduler.Scheduler

class Schedulers():
    def __init__(self,Solver, optimizer: torch.optim.Optimizer) -> None:
        self.Solver = Solver
        self.optimizer = optimizer
        
        if Solver.LR_SCHEDULER_NAME == "Step_Decay":
            self.Scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,self.Step_Decay)

        if Solver.LR_SCHEDULER_NAME == "CLR":
            self.Scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,self.CLR)

        if Solver.LR_SCHEDULER_NAME == "Linear_Warmup_With_Linear_Decay":
            self.Scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,self.Linear_Warmup_With_Linear_Decay) 

        if Solver.LR_SCHEDULER_NAME == "SGDR":
            self.Scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,self.SGDR) 

    def Step_Decay(self,iteration):
        #参考论文：《Deep residual learning for image recognition》
        if iteration < self.Solver.STEPS[0]:
            lr = self.optimizer.defaults["lr"]
        elif iteration > self.Solver.STEPS[len(self.Solver.STEPS)-1]:
            lr = self.optimizer.defaults["lr"] * (self.Solver.GAMMA ** len(self.Solver.STEPS))
        else:
            #利用二分法查找iteration位于第几个step
            Left_index = 0    #左指针
            Right_index = len(self.Solver.STEPS)-1    #右指针
            while Left_index + 1 < Right_index:                
                mid_index = (Left_index + Right_index)//2
                if iteration > self.Solver.STEPS[mid_index]:
                    Left_index = mid_index
                else:
                    Right_index = mid_index
            lr = self.optimizer.defaults["lr"] * (self.Solver.GAMMA ** Right_index)
        return lr/self.optimizer.defaults["lr"]

    def Linear_Warmup_With_Linear_Decay(self,iteration):
        if iteration <= self.Solver.MAX_ITER*0.5:
            lr = self.optimizer.defaults["lr"]*0.01 + (self.optimizer.defaults["lr"]-self.optimizer.defaults["lr"]*0.01)*iteration/(self.Solver.MAX_ITER*0.5)
            return lr/self.optimizer.defaults["lr"]
        else:
            lr = self.optimizer.defaults["lr"] - (self.optimizer.defaults["lr"]-self.optimizer.defaults["lr"]*0.01)*(iteration-self.Solver.MAX_ITER*0.5)/(self.Solver.MAX_ITER*0.5)
            return lr/self.optimizer.defaults["lr"]

    def CLR(self,iteration):
        #参考论文：《Cyclical Learning Rates for Training Neural Networks》
        if (iteration // self.Solver.CLR_STEPS) % 2 == 0:
            #学习率增加
            lr = self.optimizer.defaults["lr"]*0.01 + ((iteration%self.Solver.CLR_STEPS)/self.Solver.CLR_STEPS)*(self.optimizer.defaults["lr"]-self.optimizer.defaults["lr"]*0.01)
            return lr/self.optimizer.defaults["lr"]
        else:
            #学习率减少
            lr = self.optimizer.defaults["lr"] - ((iteration%self.Solver.CLR_STEPS)/self.Solver.CLR_STEPS)*(self.optimizer.defaults["lr"]-self.optimizer.defaults["lr"]*0.01)
            return lr/self.optimizer.defaults["lr"]

    def SGDR(self,iteration):
        #参考论文：《SGDR: Stochastic Gradient Descent with Warm Restarts》
        T_0 = self.Solver.MAX_ITER*0.5
        T_cur = (iteration % int(self.Solver.MAX_ITER*0.5))+1
        lr = 0.001 * self.optimizer.defaults["lr"] + 0.5*(self.optimizer.defaults["lr"]-0.001*self.optimizer.defaults["lr"])*(1+math.cos(T_cur*math.pi/T_0))
        return lr/self.optimizer.defaults["lr"]

# ---------------------------------------------------------------------------- #
# build loss function
# ---------------------------------------------------------------------------- #
def build_Calibration_lossFun(cfg,model = None):
    if model == None:
        if cfg.CALIBRATION_SOLVER.LOSS_FUN == "CrossEntropy":
            return torch.nn.CrossEntropyLoss()
        if cfg.CALIBRATION_SOLVER.LOSS_FUN == "LeastSquare":
            return LeastSquare_Fun
        if cfg.CALIBRATION_SOLVER.LOSS_FUN == "KDEloss":
            return KDE_lossFun(cfg)
    else:
        if cfg.CALIBRATION_SOLVER.LOSS_FUN == "Dirichlet_Fun":
            lossmodel = Dirichlet_Fun(cfg,model)
            return lossmodel.lossComput

def LeastSquare_Fun(X,Y):
    Sum = 0.
    for i in range(len(X)):
        Sum = Sum + sum([(X[i][j]-Y[i][j])**2 for j in range(len(X[i]))])
    return Sum/len(X)

class Dirichlet_Fun():
    def __init__(self,cfg,model) -> None:
        self.W = model.lin.weight
        self.b = model.lin.bias
        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss()
        self.cfg = cfg

    def lossComput(self,X,Y):
        loss = self.CrossEntropyLoss(X,Y)
        W_sum = 0.
        k = self.cfg.MODEL.OUTPUT_NUM_ClASSES
        for i in range(k):
            for j in range(k):
                if i != j:
                    W_sum = W_sum + self.W[i][j]**2
        bias_sum = sum([bj**2 for bj in self.b])
        lamda = 0.001
        miu = 0.001
        #loss = loss + lamda/(k*(k-1))*W_sum + miu/k*bias_sum
        loss = loss + k*W_sum + miu*bias_sum
        return loss

class KDE_lossFun(nn.Module):
    def __init__(self,cfg) -> None:
        super().__init__()
        self.class_num = cfg.CALIBRATION_MODEL.NUM_CLASS
        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss()

    def get_classwise_confidence_and_predict(self):
        #获取confidence与预测类别
        classwise_confidence_list = []
        for j in range(self.class_num):
            tmp_list = []
            for i in range(len(self.z_list)):
                tmp_list.append(self.z_list[i][j])
            classwise_confidence_list.append(torch.stack(tmp_list))
        return torch.stack(classwise_confidence_list)

    def comput_pi_and_p_accelerate(self):
        '''
        p:代表每个样本发生的概率,多少个样本就有多少个p
        pi代表的是每个样本置信度标签(一个长度为类别数的向量),多少个样本就有多少个pi
        加速推理版(矩阵运算)
        '''
        confidence_A = self.classwise_confidence_list.permute(1,0).unsqueeze(dim=2).repeat(1,1,self.n)
        confidence_B = self.classwise_confidence_list.unsqueeze(dim=0).repeat(self.n,1,1)
        Deff = confidence_A - confidence_B

        del confidence_A
        del confidence_B

        #计算核函数
        h = sum(self.h_list)/len(self.h_list)
        He_result = (1/h)*(35/32)*torch.pow((1-(Deff/h)**2),3)

        del Deff

        He_result = relu(He_result)    #把小于0的值置0

        p_tensor1 = torch.prod(h*He_result,dim=1).detach().clone()
        p_tensor = p_tensor1.fill_diagonal_(0)   #计算连乘
        pi_tensor1 = torch.prod(He_result,dim=1).detach().clone()
        pi_tensor = pi_tensor1.fill_diagonal_(0)    #计算连乘

        del He_result

        pi_sum = pi_tensor.sum(dim=1).unsqueeze(1).repeat(1,10) + torch.finfo(torch.float64).eps

        p = (torch.sum(p_tensor,dim=1)/self.n)

        del p_tensor

        pi = []
        for i in range(self.class_num):
            sumIndex = torch.where(self.label_list==i)[0]
            tmp_pi = pi_tensor[:,sumIndex].sum(dim=1)
            pi.append(tmp_pi)
        pi = (torch.stack(pi,dim=1)/pi_sum)

        del pi_tensor
        torch.cuda.empty_cache()

        return p,pi

    def forward(self,z_list,label_list):
        crossEntry = self.CrossEntropyLoss(z_list,label_list)
        self.label_list = label_list
        self.z_list = softmax(z_list,dim=1,dtype=torch.float64)
        
        self.classwise_confidence_list= self.get_classwise_confidence_and_predict()
        self.n = len(self.classwise_confidence_list[0])
        self.sigma = []
        for confidence in self.classwise_confidence_list:
            sigma_list = []
            for conf in confidence:
                if conf < 1.1:
                    sigma_list.append(conf)
            sigma_tensor = torch.stack(sigma_list)
            self.sigma.append(torch.std(sigma_tensor))
        self.h_list = [1.06*sigma*math.pow(self.n,-(1/5)) for sigma in self.sigma]    #带宽

        #计算ECE
        p,pi = self.comput_pi_and_p_accelerate()
        ECE = 0.
        for i in range(self.n):
            confidence = self.classwise_confidence_list[:,i]
            ECE = ECE + torch.norm(confidence-pi[i],1)#*p[i]    ##采用1范数

        ECE = ECE/self.n
        print("KDE_ECE:",ECE)
        print("CrossEntry",crossEntry)
        return ECE + crossEntry

