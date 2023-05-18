import torch
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
sys.path.insert(0,parentdir) 
from customKing.utils.env import Set_seed
from customKing.config.config import get_cfg
from customKing.modeling.meta_arch.build import build_model
from utils import doTrain,doTest

def main():
        Set_seed(seed=20)
        cfg = get_cfg()
        model = build_model(cfg)
        print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))

        if cfg.MODEL.JUST_EVAL:
            if cfg.SOLVER.IS_PARALLEL:
                pretrained_dict = torch.load(cfg.MODEL.PREWEIGHT).module.state_dict()
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model.load_state_dict(pretrained_dict)
            else:
                    model = torch.load(cfg.MODEL.PREWEIGHT)
            DoTest = doTest(cfg)
            DoTest.do_test(model)
        else:
            if cfg.MODEL.PRE_WEIGHT:
                if cfg.SOLVER.IS_PARALLEL:
                    pretrained_dict = torch.load(cfg.MODEL.PREWEIGHT).module.state_dict() 
                    model_dict = model.state_dict()
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    model.load_state_dict(pretrained_dict)
                else:
                     model = torch.load(cfg.MODEL.PREWEIGHT)
            DoTrain = doTrain(cfg)
            DoTrain.do_train(model)
            print("finished training!")
        
if __name__ == "__main__":
        main()