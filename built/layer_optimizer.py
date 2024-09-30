from engine.optimizer import BaseSeperateLayer
from typing import Iterator, List, Dict, Union
from torch.nn import Module

class SeperateLayerParams(BaseSeperateLayer):
    def __init__(self, model: Module):
        super().__init__(model)

    def create_ParamSequence(self, layer_wise: bool, lr: float) -> Union[Iterator, List[Dict]]:
        if not layer_wise: return self.model.parameters()

        params = [
            {'params': self.model.features.parameters()},
            {'params': self.model.norm.parameters()},
            {'params': self.model.head.parameters(), 'lr': lr * 10}
        ]
        return params