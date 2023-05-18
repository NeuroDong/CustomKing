from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# ----------------------------------------------------------------------------- #
# Config definition
# ----------------------------------------------------------------------------- #

_C = CN()
# ---------------------------------------------------------------------------- #
# Classification model config
# ---------------------------------------------------------------------------- #
_C.DATASETS = CN()
_C.DATASETS.TRAIN = "Cifar10_train"    #train dataset
_C.DATASETS.VALID = "Cifar10_valid"    #valid dataset
_C.DATASETS.TEST = "Cifar10_test"      #test dataset

_C.MODEL = CN()
_C.MODEL.META_ARCHITECTURE = "Resnet20"  #select classification model
_C.MODEL.OUTPUT_NUM_ClASSES = 10    #set class num
_C.MODEL.INPUT_IMAGESIZE = (32,32)    #set image size
_C.MODEL.DEVICE = "cuda:0"     #select device
_C.MODEL.JUST_EVAL = False    
_C.MODEL.PRE_WEIGHT = False    
_C.MODEL.OUTPUT_DIR = "output/"+_C.DATASETS.TRAIN[:-6]+"/"+_C.MODEL.META_ARCHITECTURE+"/metric.json"    #Path to save training log files and network weights files
_C.MODEL.PREWEIGHT = r" "    #The path of saving the pretrain weight 

_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER = "SGD"    #select optimizer，see：customKing\solver\build.py
_C.SOLVER.BATCH_SIZE = 128    #Set batch_size
_C.SOLVER.SHUFFLE = True    
_C.SOLVER.NUM_WORKERS = 8    #the num workers of the Dataloader
_C.SOLVER.IS_PARALLEL = False   #Whether to use multiple GPUs for training

_C.SOLVER.LR_SCHEDULER_NAME = "Step_Decay"     #select lr_scheduler，see：customKing\solver\build.py
_C.SOLVER.START_ITER = 0    
_C.SOLVER.MAX_EPOCH = 200    
_C.SOLVER.MAX_ITER = 64000    
_C.SOLVER.BASE_LR = 0.1    
_C.SOLVER.MOMENTUM = 0.9  
_C.SOLVER.NESTEROV = False  
_C.SOLVER.WEIGHT_DECAY = 0.0001     
_C.SOLVER.GAMMA = 0.1    #if using Step_Decay，the lr after decay is BASE_LR * GAMMA
# The iteration number to decrease learning rate by GAMMA.
_C.SOLVER.STEPS = (32000,48000)    #Set the decay step size, which must be smaller than the training MAX_ITER
_C.SOLVER.CLR_STEPS = 2000     #if using CLR lr_scheduler, the config need to set.
_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"