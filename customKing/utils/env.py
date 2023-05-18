import torch
import numpy as np
import random
import os

__all__ = ["Set_seed"]

def Set_seed(seed=20):
    # 设置种子，方便复现
    SEED = 20
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False