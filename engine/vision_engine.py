import torch
from torchvision.transforms import CenterCrop, Resize, RandomResizedCrop, Compose
import torch.nn as nn
from torch.cuda.amp import GradScaler
from dataset.basedataset import ImageDatasets
from dataset.transforms import CenterCropAndResize, SPATIAL_TRANSFORMS, create_AugTransforms
from torch.utils.data import DistributedSampler
from utils.logger import SmartLogger
from engine.optimizer import create_Optimizer
from engine.scheduler import create_Scheduler
from models.losses.loss import create_Lossfn
from engine.procedure.train import Trainer
from functools import reduce, partial
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.plots import colorstr
from structure.sampler import OHEMImageSampler
from built.layer_optimizer import SeperateLayerParams
import time
import os
import datetime
import yaml
from copy import deepcopy
from models.ema import ModelEMA
from built.class_augmenter import ClassWiseAugmenter
from models import get_model
from dataset.dataprocessor import SmartDataProcessor
from utils.average_meter import AverageMeter

__all__ = ['yaml_load', 'CenterProcessor','increment_path', 'check_cfgs_classification']


def yaml_load(file='data.yaml'):
    # Single-line safe yaml loading
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  
            if not os.path.exists(p):  
                break
        path = Path(p)


    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  

    return path

def check_cfgs_face(cfgs):
    model_cfg = cfgs['model']
    data_cfg = cfgs['data']
    # num_classes
    train_classes = [x for x in os.listdir(Path(data_cfg['root'])/'train') if not (x.startswith('.') or x.startswith('_'))]
    nc = model_cfg['head'][next(iter(model_cfg['head'].keys()))]['num_class']
    assert nc == len(train_classes), 'num_classes in model should be equal to classes in data folder'
    # pair_txt
    assert os.path.isfile(data_cfg['val']['pair_txt']), 'make sure pair_txt exists'
    from engine.faceX.evaluation import Evaluator
    with open(data_cfg['val']['pair_txt']) as f:
        pair_list = [line.strip() for line in f.readlines()]
    Evaluator.check_nps(pair_list)

def check_cfgs_classification(cfgs):
    model_cfg = cfgs['model']
    data_cfg = cfgs['data']
    hyp_cfg = cfgs['hyp']
    # num_classes
    train_classes = [x for x in os.listdir(Path(data_cfg['root'])/'train') if not (x.startswith('.') or x.startswith('_'))]
    assert model_cfg['num_classes'] == len(train_classes), 'num_classes in model should be equal to classes in data folder'
    # model
    assert model_cfg['name'].split('-')[0] in {'torchvision', 'custom'}, 'if from torchvision, torchvision-ModelName; if from your own, custom-ModelName'
    if model_cfg['kwargs'] and model_cfg['pretrained']:
        for k in model_cfg['kwargs'].keys():
            if k not in {'dropout','attention_dropout', 'stochastic_depth_prob'}: raise KeyError('set kwargs except dropout, pretrained must be False')
    assert (model_cfg['pretrained'] and ('normalize' in data_cfg['train']['augment']) and ('normalize' in data_cfg['val']['augment'])) or \
           (not model_cfg['pretrained']) and ('normalize' not in data_cfg['train']['augment']) and ('normalize' not in data_cfg['val']['augment']),\
           'if not pretrained, normalize is not necessary, or normalize is necessary'
    # loss
    assert reduce(lambda x, y: int(x) + int(y[0]), list(hyp_cfg['loss'].values())) == 1, 'ce or bce'
    # optimizer
    assert hyp_cfg['optimizer'][0] in {'sgd', 'adam', 'sam'}, 'optimizer choose sgd adam sam'
    # scheduler
    assert hyp_cfg['scheduler'] in {'linear', 'cosine', 'linear_with_warm', 'cosine_with_warm'}, 'scheduler support linear cosine linear_with_warm and cosine_with_warm'
    assert hyp_cfg['warm_ep'] >= 0 and isinstance(hyp_cfg['warm_ep'], int) and hyp_cfg['warm_ep'] < hyp_cfg['epochs'], 'warm_ep not be negtive, and should smaller than epochs'
    if hyp_cfg['warm_ep'] == 0: assert hyp_cfg['scheduler'] in {'linear', 'cosine'}, 'no warm, linear or cosine supported'
    if hyp_cfg['warm_ep'] > 0: assert hyp_cfg['scheduler'] in {'linear_with_warm', 'cosine_with_warm'}, 'with warm, linear_with_warm or cosine_with_warm supported'
    # strategy
    # focalloss
    if hyp_cfg['strategy']['focal'][0]: assert hyp_cfg['loss']['bce'], 'focalloss only support bceloss'
    # ohem
    if hyp_cfg['strategy']['ohem'][0]: assert not hyp_cfg['loss']['bce'][0], 'ohem not support bceloss'
    # mixup
    mixup, mixup_milestone = hyp_cfg['strategy']['mixup']
    assert mixup >= 0 and mixup <= 1 and isinstance(mixup_milestone, list), 'mixup_ratio[0,1], mixup_milestone be list'
    mix0, mix1 = mixup_milestone
    assert isinstance(mix0, int) and isinstance(mix1, int) and mix0 < mix1, 'mixup must List[int], start < end'
    hyp_cfg['strategy']['mixup'] = [mixup, mixup_milestone]
    # progressive learning
    if hyp_cfg['strategy']['prog_learn']: assert mixup > 0 and data_cfg['train']['aug_epoch'] >= mix1, 'if progressive learning, make sure mixup > 0, and aug_epoch >= mix_end'
    # imgsz
    assert get_imgsz(cfgs['data']['train']['augment']) == get_imgsz(cfgs['data']['val']['augment']), 'imgsz should be same in training and inference'

def get_imgsz(augment: dict):
    augments = create_AugTransforms(augment)
    for a in augments.transforms[::-1]:
        if type(a) in SPATIAL_TRANSFORMS and hasattr(a, 'size'):
            if type(a.size) is int: return (a.size, a.size)
            elif type(a.size) in [tuple, list]: return tuple(a.size)
            else: raise ValueError('size be int, tuple or list')

class CenterProcessor:
    def __init__(self, cfgs: dict, rank: int, project: str = None, train: bool = True, opt = None):
        log_filename = Path(project) / "log{}.log".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) if project is not None else None
        self.project = project
        if rank in {-1, 0} and train:
            project.mkdir(parents=True, exist_ok=True)

        self.cfgs = cfgs
        self.model_cfg = cfgs['model']
        self.data_cfg = cfgs['data']
        self.hyp_cfg = cfgs['hyp']
        self.opt = opt
        self.imgsz = get_imgsz(cfgs['data']['train']['augment'])

        # task
        self.face = self.model_cfg['task'] == 'face'

        # rank
        self.rank: int = rank
        # device
        if rank != -1:
            device = torch.device('cuda', rank)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device: torch.device = device

        # logger
        self.logger = SmartLogger(filename=log_filename, level=1) if rank in {-1,0} else None
        if self.logger is not None and rank in {-1, 0} and train:
            self.logger.console(cfgs) # output configs

        # model processor
        self.model_processor = get_model(self.model_cfg, self.logger, rank)
        self.model_processor.model.to(device)
        # data processor
        self.data_processor = SmartDataProcessor(self.data_cfg, rank=rank, project=project)

        # loss
        loss_choice: str = 'ce' if self.hyp_cfg['loss']['ce'] else 'bce'
        self.loss_choice = loss_choice
        if not self.face:
            if train:
                self.lossfn = create_Lossfn(loss_choice)() \
                    if loss_choice == 'bce' \
                    else create_Lossfn(loss_choice)(label_smooth = self.hyp_cfg['label_smooth'])
            self.thresh = self.hyp_cfg['loss']['bce'][1] if loss_choice == 'bce' else 0

            # add label_transforms
            if loss_choice == 'bce':
                self.data_processor.train_dataset.label_transforms = \
                    partial(ImageDatasets.set_label_transforms,
                            num_classes = self.model_cfg['num_classes'],
                            label_smooth = self.hyp_cfg['label_smooth'])
                self.data_processor.val_dataset.label_transforms = \
                    partial(ImageDatasets.set_label_transforms,
                            num_classes=self.model_cfg['num_classes'],
                            label_smooth=self.hyp_cfg['label_smooth'])
                # bce not support self.sampler
                self.sampler = None
                self.data_processor.train_dataset.multi_label = self.hyp_cfg['loss']['bce'][2]
                self.data_processor.val_dataset.multi_label = self.hyp_cfg['loss']['bce'][2]
            # ohem
            elif self.hyp_cfg['strategy']['ohem'][0]: self.sampler = OHEMImageSampler(*self.hyp_cfg['strategy']['ohem'][1:])
            else: self.sampler = None

        else: self.lossfn = create_Lossfn(loss_choice)(label_smooth = self.hyp_cfg['label_smooth'])

        if train and not self.face:
            # distributions sampler
            self.dist_sampler = self._distributions_sampler()

            # progressive learning
            self.prog_learn = self.hyp_cfg['strategy']['prog_learn']
            # mixup change node
            if self.prog_learn: self.mixup_chnodes = torch.linspace(*self.hyp_cfg['strategy']['mixup'][-1], 3,dtype=torch.int32).round_().tolist()

            # focalloss hard
            if loss_choice == 'bce' and self.hyp_cfg['strategy']['focal'][0]:
                self.focal = create_Lossfn('focal')(gamma=self.hyp_cfg['strategy']['focal'][2], alpha= self.hyp_cfg['strategy']['focal'][1])
            else:
                self.focal = None

        # ema
        if train: self.ema = ModelEMA(self.model_processor.model) if rank in {-1, 0} else None

        self.loss_meter = AverageMeter()

    def set_optimizer_momentum(self, momentum) -> None:
        self.optimizer.param_groups[0]['momentum'] = momentum

    def _distributions_sampler(self):
        d = {}

        d['uniform'] = torch.distributions.uniform.Uniform(low=0, high=1)
        d['beta'] = None

        return d

    def auto_mixup(self, mixup: float, epoch:int, milestone: list) -> float:
        if mixup == 0 or epoch < milestone[0] or self.dist_sampler['beta'] is None : return 0
        else:
            mix_prob = self.dist_sampler['uniform'].sample()
            lam = self.dist_sampler['beta'].sample().to(self.device) if mix_prob < mixup else 0

            return lam

    def auto_prog(self, epoch: int):
        def create_AugSequence(train_augs : list, size):
            sequence = []
            for i, m in enumerate(train_augs):
                if isinstance(m, CenterCrop):
                    if i + 1 < len(train_augs) and (not isinstance(train_augs[i + 1], Resize) or not isinstance(train_augs[i + 1], RandomResizedCrop)):
                        sequence.extend([m, Resize(size)])
                    else:
                        sequence.append(m)
                elif isinstance(m, Resize):
                    sequence.append(Resize(size))
                elif isinstance(m, CenterCropAndResize):
                    m[-1] = Resize(size)
                    sequence.append(m)
                elif isinstance(m, RandomResizedCrop):
                    m.size = (size, size)
                    sequence.append(m)
                else:
                    sequence.append(m)

            return sequence
        chnodes = self.mixup_chnodes
        # mixup, divide mixup_milestone into 2 parts in default, alpha from 0.1 to 0.2
        if epoch in chnodes:
            alpha = self.mixup_chnodes.index(epoch) * 0.1
            if alpha != 0:
                self.dist_sampler['beta'] = torch.distributions.beta.Beta(alpha, alpha)
        # image resize, based on mixup_milestone
        min_imgsz = min(self.imgsz)
        imgsz_milestone = torch.linspace(int(min_imgsz * 0.5), int(min_imgsz), 3, dtype=torch.int32).tolist()

        if epoch == chnodes[0]: size = imgsz_milestone[0]
        elif epoch == chnodes[1]: size = imgsz_milestone[1]
        elif epoch == chnodes[2]: size = imgsz_milestone[2]
        else: return

        if hasattr(self.data_processor.train_dataset.transforms, 'base_transforms'):
            transforms = self.data_processor.train_dataset.transforms.base_transforms.transforms
            self.data_processor.train_dataset.transforms.base_transforms = Compose(create_AugSequence(transforms, size))
        if hasattr(self.data_processor.train_dataset.transforms, 'class_transforms') and self.data_processor.train_dataset.transforms.class_transforms is not None:
            for c, transforms in self.data_processor.train_dataset.transforms.class_transforms.items():
                self.data_processor.train_dataset.transforms.class_transforms[c] = Compose(create_AugSequence(transforms.transforms, size))
        if hasattr(self.data_processor.train_dataset.transforms, 'common_transforms') and self.data_processor.train_dataset.transforms.common_transforms is not None:
            transforms = self.data_processor.train_dataset.transforms.common_transforms.transforms
            self.data_processor.train_dataset.transforms.common_transforms = Compose(create_AugSequence(transforms, size))

    def set_sync_bn(self):
        self.model_processor.model = nn.SyncBatchNorm.convert_sync_batchnorm(module=self.model_processor.model)

    def run_classifier(self, resume = None): # train+val per epoch
        last, best = self.project / 'last.pt', self.project / 'best.pt'

        model, data_processor, scaler, device, epochs, logger, (mixup, mixup_milestone), rank, distributions_sampler, warm_ep, aug_epoch, focal, sampler, thresh = \
            self.model_processor.model, self.data_processor, \
            GradScaler(enabled = (self.device != torch.device('cpu'))), self.device, self.hyp_cfg['epochs'], \
            self.logger, self.hyp_cfg['strategy']['mixup'], self.rank, self.dist_sampler, self.hyp_cfg['warm_ep'], \
            self.data_cfg['train']['aug_epoch'], self.focal, self.sampler, self.thresh

        # data
        train_dataset, val_dataset = data_processor.train_dataset, data_processor.val_dataset
        data_sampler = None if self.rank == -1 else DistributedSampler(dataset=train_dataset)
        train_dataloader = data_processor.set_dataloader(dataset=train_dataset,
                                                         bs=self.data_cfg['train']['bs'],
                                                         nw=self.data_cfg['nw'],
                                                         pin_memory=True,
                                                         sampler=data_sampler,
                                                         shuffle=data_sampler is None,
                                                         collate_fn=train_dataset.collate_fn)

        if self.rank in {-1, 0}:
            val_dataloader = data_processor.set_dataloader(dataset=val_dataset,
                                                           bs=self.data_cfg['val']['bs'],
                                                           nw=self.data_cfg['nw'],
                                                           pin_memory=False,
                                                           shuffle=False,
                                                           collate_fn=val_dataset.collate_fn)
        else:
            val_dataloader = None

        # tell data distribution
        if rank in (-1, 0):
            ImageDatasets.tell_data_distribution(self.data_cfg['root'], logger, self.model_cfg['num_classes'])

        # optimizer
        params = SeperateLayerParams(model)
        optimizer = create_Optimizer(optimizer=self.hyp_cfg['optimizer'][0],
                                     lr=self.hyp_cfg['lr0'],
                                     weight_decay=self.hyp_cfg['weight_decay'],
                                     momentum=self.hyp_cfg['warmup_momentum'],
                                     params=params.create_ParamSequence(layer_wise=self.hyp_cfg['optimizer'][1],
                                                                        lr=self.hyp_cfg['lr0']))
        self.optimizer = optimizer
        # scheduler
        scheduler = create_Scheduler(scheduler=self.hyp_cfg['scheduler'],
                                     optimizer=optimizer,
                                     warm_ep=self.hyp_cfg['warm_ep'],
                                     epochs=self.hyp_cfg['epochs'],
                                     lr0=self.hyp_cfg['lr0'],
                                     lrf_ratio=self.hyp_cfg['lrf_ratio'])

        best_fitness = 0.
        start_epoch = 0

        # resume
        if resume is not None:
            ckp = torch.load(resume, map_location=device)
            start_epoch = ckp['epoch'] + 1
            best_fitness = ckp['best_fitness']
            if self.rank in {-1, 0}:
                self.ema.ema.load_state_dict(ckp['ema'].float().state_dict())
                self.ema.updates = ckp['updates']
            model.load_state_dict(ckp['model'])
            optimizer.load_state_dict(ckp['optimizer'])
            scheduler.load_state_dict(ckp['scheduler'])
            if device != torch.device('cpu'):
                scaler.load_state_dict(ckp['scaler'])

            if rank in (-1, 0): logger.both(f'resume: {resume}')

        if rank != -1:
            model = DDP(model, device_ids=[self.rank])

        if self.rank in {-1, 0}:

            if thresh == 0:
                print(f"{'Epoch':>10}{'GPU_mem':>10}{'train_loss':>12}{f'val_loss':>12}{'top1_acc':>12}{'top5_acc':>12}")
            else:
                print(f"{'Epoch':>10}{'GPU_mem':>10}{'train_loss':>12}{f'val_loss':>12}{'precision':>12}{'recall':>12}{'f1score':>12}")
            time.sleep(0.2)

        # total epochs
        total_epoch = epochs+warm_ep

        # trainer
        trainer = Trainer(model, train_dataloader, val_dataloader, optimizer,
                          scaler, device, total_epoch, logger, rank, scheduler, self.ema, sampler, thresh,
                          self.teacher if hasattr(self, 'teacher') else None, cfgs=self.cfgs)

        t0 = time.time()
        for epoch in range(start_epoch, total_epoch):
            # warmup set augment as val
            if epoch == 0:
                self.data_processor.set_augment('train', sequence=None)

            # change optimizer momentum from warm_moment0.8 -> momentum0.937
            if epoch == warm_ep:
                self.set_optimizer_momentum(self.hyp_cfg['momentum'])
                self.data_processor.set_augment('train', sequence=ClassWiseAugmenter(self.data_cfg['train']['augment'], self.data_cfg['train']['class_aug'], self.data_cfg['train']['common_aug']))

            # change lossfn bce -> focal
            if int(epoch-warm_ep) == 0 and self.focal is not None:
                self.lossfn = self.focal

            # weaken data augment at milestone
            self.data_processor.auto_aug_weaken(int(epoch-warm_ep), milestone=aug_epoch)
            if int(epoch-warm_ep) == aug_epoch: self.dist_sampler['beta'] = None # weaken mixup

            # progressive learning: effect on imagesz & mixup
            if self.prog_learn:
                self.auto_prog(epoch=int(epoch-warm_ep))

            # mixup epoch-wise
            lam = self.auto_mixup(mixup=mixup, epoch=int(epoch-warm_ep), milestone=mixup_milestone)

            # train for one epoch
            fitness = trainer.train_one_epoch(epoch, lam, self.lossfn)

            if rank in {-1, 0}:
                # Best fitness
                if fitness > best_fitness:
                    best_fitness = fitness

                # Save model
                final_epoch: bool = epoch + 1 == total_epoch
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': model.state_dict() if rank == -1 else model.module.state_dict(),  # deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(self.ema.ema),
                    'updates': self.ema.updates,
                    'optimizer': optimizer.state_dict(),  # optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                if device != torch.device('cpu'):
                    ckpt['scaler'] = scaler.state_dict()

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fitness:
                    torch.save(ckpt, best)
                del ckpt

                # complete
                if final_epoch:
                    logger.both(f'\nTraining complete ({(time.time() - t0) / 3600:.3f} hours)'
                                   f"\nResults saved to {colorstr('bold', self.project)}"
                                   f'\nPredict:         python visualize.py --cfgs {os.path.join(os.path.dirname(best), os.path.basename(self.opt.cfgs))} --weight {best} --badcase --class_json {self.project}/class_indices.json --ema --cam --data {data_processor.data_cfgs["root"]}/val/{colorstr("blue", "XXX_cls")}'
                                   f'\nValidate:        python validate.py --cfgs {os.path.join(os.path.dirname(best), os.path.basename(self.opt.cfgs))} --eval_topk 5 --weight {best} --ema' )

    def run_face(self, resume = None):
        model, data_processor, scaler, device, epochs, logger, rank, warm_ep, aug_epoch = self.model_processor.model, self.data_processor, \
            GradScaler(enabled = (self.device != torch.device('cpu'))), self.device, self.hyp_cfg['epochs'], self.logger, self.rank, self.hyp_cfg['warm_ep'], \
            self.data_cfg['train']['aug_epoch']

        # load for fine-tune
        if 'load_from' in self.model_cfg:
            model.load_state_dict(torch.load(self.model_cfg['load_from'], map_location='cpu')['state_dict'], strict=True)
            if rank in (-1, 0): logger.both(f'load_from: {self.model_cfg["load_from"]}')

        # data
        train_dataset, val_dataset = data_processor.train_dataset, data_processor.val_dataset
        data_sampler = None if self.rank == -1 else DistributedSampler(dataset=train_dataset)
        train_dataloader = data_processor.set_dataloader(dataset=train_dataset,
                                                         bs=self.data_cfg['train']['bs'],
                                                         nw=self.data_cfg['nw'],
                                                         pin_memory=True,
                                                         sampler=data_sampler,
                                                         shuffle=data_sampler is None,
                                                         collate_fn=train_dataset.collate_fn)
        # tell data distribution
        if self.rank in (-1, 0):
            ImageDatasets.tell_data_distribution(self.data_cfg['root'], logger, self.model_cfg['head'][next(iter(self.model_cfg['head'].keys()))]['num_class'])

        # optimizer
        params = SeperateLayerParams(model)
        optimizer = create_Optimizer(optimizer=self.hyp_cfg['optimizer'][0],
                                     lr=self.hyp_cfg['lr0'],
                                     weight_decay=self.hyp_cfg['weight_decay'],
                                     momentum=self.hyp_cfg['warmup_momentum'],
                                     params=params.create_ParamSequence(layer_wise=self.hyp_cfg['optimizer'][1],
                                                                        lr=self.hyp_cfg['lr0']))
        self.optimizer = optimizer
        # scheduler
        scheduler = create_Scheduler(scheduler=self.hyp_cfg['scheduler'],
                                     optimizer=optimizer,
                                     warm_ep=self.hyp_cfg['warm_ep'] * len(train_dataloader),
                                     epochs=self.hyp_cfg['epochs'] * len(train_dataloader),
                                     lr0=self.hyp_cfg['lr0'],
                                     lrf_ratio=self.hyp_cfg['lrf_ratio'])

        start_epoch = 0

        # resume
        if resume is not None:
            ckp = torch.load(resume, map_location=device)
            start_epoch = ckp['epoch'] + 1

            if self.rank in {-1, 0}:
                self.ema.ema.load_state_dict(ckp['ema'].float().state_dict())
                self.ema.updates = ckp['updates']
            model.load_state_dict(ckp['state_dict'])
            optimizer.load_state_dict(ckp['optimizer'])
            scheduler.load_state_dict(ckp['scheduler'])
            if device != torch.device('cpu'):
                scaler.load_state_dict(ckp['scaler'])

            if rank in (-1, 0): logger.both(f'resume: {resume}')

        if rank != -1:
            model = DDP(model, device_ids=[self.rank])

        if self.rank in {-1, 0}: time.sleep(0.2)

        # total epochs
        total_epoch = epochs + warm_ep

        # trainer
        trainer = Trainer(model, train_dataloader, None, optimizer,
                          scaler, device, total_epoch, logger, rank, scheduler, self.ema, None, None,
                          self.teacher if hasattr(self, 'teacher') else None, self.opt.print_freq, self.opt.save_freq, self.cfgs, self.opt.save_dir)

        t0 = time.time()
        for epoch in range(start_epoch, total_epoch):
            # warmup set augment as val
            if epoch == 0:
                self.data_processor.set_augment('train', sequence=None)

            # change optimizer momentum from warm_moment0.8 -> momentum0.937
            if epoch == warm_ep:
                self.set_optimizer_momentum(self.hyp_cfg['momentum'])
                self.data_processor.set_augment('train',
                                                sequence=ClassWiseAugmenter(self.data_cfg['train']['augment'],
                                                                            self.data_cfg['train']['class_aug'],
                                                                            self.data_cfg['train']['common_aug']))
            # weaken data augment at milestone
            self.data_processor.auto_aug_weaken(int(epoch-warm_ep), milestone=aug_epoch)

            # train for one epoch
            trainer.train_one_epoch_face(self.lossfn, epoch, self.loss_meter)

        if rank in (-1, 0):
            logger.both(f'\nTraining complete ({(time.time() - t0) / 3600:.3f} hours)'
                        f"\nResults saved to {colorstr('bold', self.project)}"
                        f'\nValidate:        python validate.py --cfgs {self.opt.cfgs} --weight {self.project}/{colorstr("blue", "which_weight")} --ema')
