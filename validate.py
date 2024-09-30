import os
from os.path import join as opj
import argparse
from pathlib import Path
from engine import valuate as valuate_classifier , CenterProcessor, yaml_load
import torch
from models.faceX.face_model import FaceModelLoader, FeatureExtractor
from engine.faceX.evaluation import valuate as valuate_face
from prettytable import PrettyTable
from utils.logger import SmartLogger

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
ROOT = Path(os.path.dirname(__file__))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgs', default = ' ', help = 'Configs for models, data, hyps')
    parser.add_argument('--weight', default = 'run/exp/best.pt', help='Weight path')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    parser.add_argument('--ema', action='store_true',help = 'Exponential Moving Average for model weight')
    parser.add_argument('--eval_topk', default = 5, type=int, help = 'Tell topk_acc, maybe top5, top3...')
    return parser.parse_args()

def main(opt):
    cfgs = yaml_load(opt.cfgs)
    face: bool = cfgs['model']['task']
    if not face:
        cpu = CenterProcessor(cfgs, LOCAL_RANK, train=False, opt=opt)

        # checkpoint loading
        model = cpu.model_processor.model
        if opt.ema:
            weights = torch.load(opt.weight, map_location=cpu.device)['ema'].float().state_dict()
        else:
            weights = torch.load(opt.weight, map_location=cpu.device)['model']
        model.load_state_dict(weights)

        # set val dataloader
        dataloader = cpu.data_processor.set_dataloader(cpu.data_processor.val_dataset, nw=cpu.data_cfg['nw'], bs=cpu.data_cfg['train']['bs'],
                                                       collate_fn=cpu.data_processor.val_dataset.collate_fn)

        conm_path = opj(os.path.dirname(opt.weight), 'conm.png')
        valuate_classifier(model, dataloader, cpu.device, None, False, None, cpu.logger, thresh=cpu.thresh, top_k=opt.eval_topk,
                conm_path=conm_path)
    else:
        # logger
        logger = SmartLogger(filename=None)

        # checkpoint loading
        logger.console(f'loading model, ema is {opt.ema}')
        model_loader = FaceModelLoader(model_cfg=cfgs['model'])
        model = model_loader.load_weight(model_path=opt.weight, ema=opt.ema)

        logger.console('valuating...')
        mean, std = valuate_face(model, cfgs['data'], torch.device('cuda'))
        pretty_tabel = PrettyTable(["model_name", "mean accuracy", "standard error"])
        pretty_tabel.add_row([os.path.basename(opt.weight), mean, std])

        logger.console('\n' + str(pretty_tabel))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)