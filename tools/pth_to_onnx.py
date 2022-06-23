import os
from detectron2.config import get_cfg
from detectron2.modeling import build_model
import torch

def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    #--------------------设置配置参数-----------------------------#
    cfg = get_cfg() 
    cfg.MODEL.META_ARCHITECTURE = "Resnet20" #网络模型
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
    model.eval()
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
    #model = torch.jit.script(model)

    x = torch.randn(1,3,224,224).cuda().float()
    torch.onnx.export(model,x,"deploy_output/"+cfg.MODEL.META_ARCHITECTURE+".onnx",export_params=False)

if __name__ == "__main__":
    main()