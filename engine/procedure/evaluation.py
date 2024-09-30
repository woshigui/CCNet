import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm
from typing import Callable, Optional
from torchmetrics import Precision, Recall, F1Score
from torch import Tensor
import itertools
import matplotlib.pyplot as plt
import numpy as np
from typing import Sequence


__all__ = ['valuate', 'predict_images']

class ConfusedMatrix:
    def __init__(self, nc: int):
        self.nc = nc
        self.mat = None

    def update(self, gt: Tensor, pred: Tensor):
        if self.mat is None: self.mat = torch.zeros((self.nc, self.nc), dtype=torch.int64, device = gt.device)

        idx = gt * self.nc + pred
        self.mat += torch.bincount(idx, minlength=self.nc).reshape(self.nc, self.nc)

    def save_conm(self, cm: np.ndarray, classes: Sequence, save_path: str, cmap=plt.cm.cool):
        ax = plt.gca()
        ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.colorbar()
        tick_marks = [x for x in range(len(classes))]
        plt.xticks(tick_marks, classes, rotation=0, fontsize=10)
        plt.yticks(tick_marks, classes, fontsize=10)
        fmt = '.2f'
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="black")
        plt.tight_layout()
        plt.ylabel('GT', fontsize=12)
        plt.xlabel('Predict', fontsize=12)
        ax.xaxis.set_label_position('top')
        plt.gcf().subplots_adjust(top=0.9)
        plt.savefig(save_path)

def valuate(model: nn.Module, dataloader, device: torch.device, pbar, is_training: bool = False, lossfn: Optional[Callable] = None, logger = None, thresh: float = 0, top_k: int = 5, conm_path: str = None):

    assert thresh == 0 or thresh > 0 and thresh < 1, 'softmax时thresh为0 bce时0 < thresh < 1'
    # eval mode
    model.eval()

    n = len(dataloader)  # number of batches
    action = 'validating'
    desc = f'{pbar.desc[:-36]}{action:>36}' if pbar else f'{action}'
    bar = tqdm(dataloader, desc, n, not is_training, bar_format='{l_bar}{bar:10}{r_bar}', position=0)
    pred, targets, loss = [], [], 0
    with torch.no_grad(): # w/o this op, computation graph will be save
        with autocast(enabled=(device != torch.device('cpu'))):
            for images, labels in bar:
                images, labels = images.to(device, non_blocking = True), labels.to(device)
                y = model(images)
                if thresh == 0:
                    pred.append(y.argsort(1, descending=True)[:, :top_k])
                    targets.append(labels)
                else:
                    pred.append(y.sigmoid())
                    hard_labels = labels.round()
                    hard_labels = torch.where(hard_labels == 1, hard_labels, 0)
                    targets.append(hard_labels)
                if lossfn:
                    loss += lossfn(y, labels)

    loss /= n
    pred, targets = torch.cat(pred), torch.cat(targets)
    if not is_training and thresh == 0:
        conm = ConfusedMatrix(len(dataloader.dataset.class_indices))
        conm.update(targets, pred[:, 0])
        conm.save_conm(conm.mat.detach().cpu().numpy(), dataloader.dataset.class_indices, conm_path if conm_path is not None else 'conm.png')

    if thresh == 0:
        correct = (targets[:, None] == pred).float()
        acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) accuracy
        top1, top5 = acc.mean(0).tolist()
        if not is_training: logger.console(f'{"name":<15}{"nums":>8}{"top1":>10}{f"top{top_k}":>10}')
        for i, c in enumerate(dataloader.dataset.class_indices):
            acc_i = acc[targets == i]
            top1i, top5i = acc_i.mean(0).tolist()
            if not is_training: logger.console(f'{c:<15}{acc_i.shape[0]:>8}{top1i:>10.3f}{top5i:>10.3f}')
            else: logger.log(f'{c:<8}{acc_i.shape[0]:>8}{top1i:>10.3f}{top5i:>10.3f}')
        if not is_training: logger.console(f'{"    ":<15}{acc.shape[0]:>8}{top1:>10.3f}{round(top5, 3):>10.3f}')
    else:
        num_classes = len(dataloader.dataset.class_indices)
        precisioner = Precision(task='multilabel', threshold=thresh, num_labels=num_classes, average=None).to(device)
        recaller = Recall(task='multilabel', threshold=thresh, num_labels=num_classes, average=None).to(device)
        f1scorer = F1Score(task='multilabel', threshold=thresh, num_labels=num_classes, average=None).to(device)

        precision = precisioner(pred, targets)
        recall = recaller(pred, targets)
        f1score = f1scorer(pred, targets)

        if not is_training: logger.console(f'{"name":<8}{"nums":>8}{"precision":>10}{"recall":>10}{"f1-score":>10}')
        cls_numbers = torch.bincount(torch.argmax(targets, dim=1), minlength=pred.shape[-1]).tolist()
        for i, c in enumerate(dataloader.dataset.class_indices):
            if not is_training: logger.console(f'{c:<8}{cls_numbers[i]:>8}{precision[i].item():>10.3f}{recall[i].item():>10.3f}{f1score[i].item():>10.3f}')
            else: logger.log(f'{c:<8}{cls_numbers[i]:>8}{precision[i].item():>15.3f}{recall[i].item():>10.3f}{f1score[i].item():>10.3f}')
        if not is_training: logger.console(
            f'mprecision:{precision.mean().item():.3f}, mrecall:{recall.mean().item():.3f}, mf1-score:{f1score.mean().item():.3f},')

    if pbar and thresh == 0:
        pbar.desc = f'{pbar.desc[:-36]}{loss:>12.3g}{top1:>12.3g}{top5:>12.3g}'
    elif pbar and thresh > 0:
        pbar.desc = f'{pbar.desc[:-36]}{loss:>12.3g}{precision.mean().item():>12.3g}{recall.mean().item():>12.3g}{f1score.mean().item():>12.3g}'

    if lossfn:
        if thresh == 0: return top1, top5, loss
        else: return precision.mean().item(), recall.mean().item(), f1score.mean().item(), loss
    else:
        if thresh == 0: return top1, top5
        else: return precision.mean().item(), recall.mean().item(), f1score.mean().item(),

def predict_images(model, dataloader, root, device, visual_path, class_indices: dict, logger, class_head: str, badcase: bool, is_cam: bool, no_annotation: bool, target_class: Optional[str] = None):

    import glob
    from utils.plots import Annotator
    from functools import reduce
    import platform
    import shutil
    import os
    import torch.nn.functional as F
    import cv2

    os.makedirs(visual_path, exist_ok=True)

    # eval mode
    model.eval()
    n = len(dataloader)

    # cam
    if is_cam:
        from utils.cam import ClassActivationMaper
        cam = ClassActivationMaper(model, method='gradcam', device=device, transforms=dataloader.dataset.transforms)

    image_postfix_table = dict() # use for badcase
    for i, (img, inputs, img_path) in enumerate(dataloader):
        img = img[0]
        img_path = img_path[0]

        if is_cam:
            cam_image = cam(image=img, input_tensor=inputs, dsize=img.size)
            cam_image = cv2.resize(cam_image, img.size, interpolation=cv2.INTER_LINEAR)

        # system
        if platform.system().lower() == 'windows':
            annotator = Annotator(img, font=r'C:/WINDOWS/FONTS/SIMSUN.TTC') # windows
        else:
            annotator = Annotator(img) # linux

        # transforms
        inputs = inputs.to(device)
        # forward
        logits = model(inputs).squeeze()

        # post process
        if class_head == 'ce':
            probs = F.softmax(logits, dim=0)
        else:
            probs = F.sigmoid(logits)

        top5i = probs.argsort(0, descending=True)[:5].tolist()

        text = '\n'.join(f'{probs[j].item():.2f} {class_indices[j]}' for j in top5i)

        if not no_annotation:
            annotator.text((32, 32), text, txt_color=(0, 0, 0))


        if badcase:  # Write to file
            os.makedirs(os.path.join(visual_path, 'labels'), exist_ok=True)
            image_postfix_table[os.path.basename(os.path.splitext(img_path)[0] + '.txt')] = os.path.splitext(img_path)[1]
            with open(os.path.join(visual_path, 'labels', os.path.basename(os.path.splitext(img_path)[0] + '.txt')), 'a') as f:
                f.write(text + '\n')

        logger.console(f"[{i+1}|{n}] " + os.path.basename(img_path) +" " + reduce(lambda x,y: x + " "+ y, text.split()))

        if is_cam:
            img = np.hstack([cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), cam_image])
            cv2.imwrite(os.path.join(visual_path, os.path.basename(img_path)), img)
        else: img.save(os.path.join(visual_path, os.path.basename(img_path)))

    if badcase:
        cls = root.split('/')[-1] if target_class is None else target_class
        assert cls in class_indices.values(), 'Either specify the correct category through target_class or the category of the folder name itself'
        os.makedirs(os.path.join(visual_path, 'bad_case'), exist_ok=True)
        for txt in glob.glob(os.path.join(visual_path, 'labels', '*.txt')):
            with open(txt, 'r') as f:
                if f.readlines()[0].split()[1] != cls:
                    try:
                        shutil.move(os.path.join(visual_path, os.path.basename(txt).replace('.txt', image_postfix_table[os.path.basename(txt)])), os.path.dirname(txt).replace('labels','bad_case'))
                    except FileNotFoundError:
                        print(f'FileNotFoundError->{txt}')
