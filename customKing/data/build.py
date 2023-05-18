from .catalog import DatasetCatalog
from torch.utils.data import DataLoader

def get_dataset_dicts(name):
    assert isinstance(name, str)
    dataset = DatasetCatalog.get(name) 
    return dataset

def build_loader(cfg,dataset):
    return DataLoader(dataset,batch_size=cfg.SOLVER.BATCH_SIZE,shuffle=cfg.SOLVER.SHUFFLE,num_workers=cfg.SOLVER.NUM_WORKERS)