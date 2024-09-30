from copy import deepcopy
import torch.nn as nn
import math

def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    return model.module if is_parallel(model) else model

class ModelEMA:

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  
        self.updates = updates  
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        self.updates += 1
        d = self.decay(self.updates)

        msd = de_parallel(model).state_dict()  
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:  
                v *= d
                v += (1 - d) * msd[k].detach()
