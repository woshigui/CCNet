import torch
import torch.nn as nn
from typing import Callable
from functools import wraps
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    'bce',
    'ce',
    'focal',
    'joint_ce_center',
    'create_Lossfn',
    'list_lossfns',
]

LOSS = {}

def register_loss(fn: Callable):
    key = fn.__name__
    if key in LOSS:
        raise ValueError(f"An entry is already registered under the name '{key}'.")
    LOSS[key] = fn
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T: float):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s: Tensor, y_t: Tensor):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)
        return loss

@register_loss
def bce():
    return nn.BCEWithLogitsLoss()

@register_loss
def ce(label_smooth: float = 0.):
    return nn.CrossEntropyLoss(label_smoothing=label_smooth)

@register_loss
def focal(gamma=1.5, alpha=0.25):
    return FocalLoss(loss_fcn=nn.BCEWithLogitsLoss(), alpha=alpha, gamma=gamma)

class CenterLoss(nn.Module):
    """
    Center Loss: https://ydwen.github.io/papers/WenECCV16.pdf
    """
    def __init__(self, num_classes: int, feat_dim: int, device: torch.device):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, features: Tensor, labels: Tensor):

        centers_batch = self.centers[labels]  # (batch_size, feat_dim)
        loss = F.mse_loss(features, centers_batch)
        return loss

class JointLoss(nn.Module):
    def __init__(self, ce_loss: nn.Module, center_loss: nn.Module, lambda_center: float = 1.0):
        super(JointLoss, self).__init__()
        self.ce_loss = ce_loss
        self.center_loss = center_loss
        self.lambda_center = lambda_center

    def forward(self, preds: Tensor, features: Tensor, labels: Tensor):
        ce = self.ce_loss(preds, labels)
        center = self.center_loss(features, labels)
        return ce + self.lambda_center * center

@register_loss
def joint_ce_center(num_classes: int, feat_dim: int, device: torch.device, label_smooth: float = 0.0, lambda_center: float = 1.0):
    ce = ce(label_smooth=label_smooth)
    center = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, device=device)
    joint_loss = JointLoss(ce_loss=ce, center_loss=center, lambda_center=lambda_center)
    return joint_loss

def create_Lossfn(lossfn: str, **kwargs):
    lossfn = lossfn.strip()
    if lossfn not in LOSS:
        raise ValueError(f"Loss function '{lossfn}' is not registered. Available losses: {list_lossfns()}")
    return LOSS[lossfn](**kwargs)

def list_lossfns():
    lossfns = [k for k, v in LOSS.items()]
    return sorted(lossfns)
