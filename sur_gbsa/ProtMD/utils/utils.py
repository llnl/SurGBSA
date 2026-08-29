import torch

import os
import logging
import argparse
import random
import numpy as np
from pathlib import Path

# def parse_args():
def get_parser():

    parser = argparse.ArgumentParser(description="Build and train Molformer.")
    # set pretraining hyper-parameters
    parser.add_argument('--ordering', type=bool, default=False, help='Whether to use the snapshot ordering task.')
    parser.add_argument('--noise', type=bool, default=True, help='Whether to add noise during pretraining.')#(derek) does this not cause a bug?
    parser.add_argument('--prompt', type=str, default='1', help='Whether to use the snapshot ordering task.')
    parser.add_argument('--max_sample', type=int, default=64, help='The default number of pre-train samples.')

    # set fine-tune hyper-parameters
    parser.add_argument('--data', type=str, default='lba', choices=['lba', 'lep'])
#    parser.add_argument('--unknown', type=bool, default=True, help='Whether to use docking structures.')
    parser.add_argument('--unknown', default=False, action='store_true', help='Whether to use docking structures.')
    parser.add_argument('--pretrain', type=str, default='', help='Whether to load the pretrained model weights.')
    parser.add_argument('--linear_probe', default=False, action='store_true')

    # set model hyper-parameters
    parser.add_argument('--num_nearest', type=int, default=32, help='The default number of nearest neighbors.')
    parser.add_argument('--tokens', type=int, default=100, help='The default number of atom classes.')
    parser.add_argument('--depth', type=int, default=6, help='Number of stacked layers.')
    parser.add_argument('--dim', type=int, default=128, help='Dimension of features.')
    parser.add_argument('--dropout', type=float, default=0.15, help='Dropout rate.')
    parser.add_argument('--max_len', type=int, default=10000, help='Maximum number of nodes for the input graph.')

    # set training details
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--split_ratio', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epoch.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate.')
    parser.add_argument('--lr_factor', type=float, default=0.6, help='factor for learning rate scheduler')
    parser.add_argument('--lr_patience', type=int, default=10, help='patience for learning rate scheduler') # this was set to 5 in the code and it seems to mess up training in that its way too aggressive. The ProtMD paper also uses 10..
    parser.add_argument('--min_lr', type=float, default=5e-7, help='The minimum learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-10, metavar='N', help='Timing experiment')
    parser.add_argument('--num_workers', type=int, default=8, help="number of workers to use for dataloading")
    parser.add_argument('--noise_scale', type=float, default=10)

    # set training environment
    parser.add_argument('--split', type=str, default='30', choices=['30', '60'])
    parser.add_argument('--data_dir', type=str, default='data/md/', help='Path for loading data.')
    parser.add_argument('--gpu', type=str, default='0', help='Index for GPU')
    parser.add_argument('--save_path', default='save/', help='Path to save the model and the logger.')
    parser.add_argument('--resume_training', action="store_true")
    parser.add_argument('--resume-ckpt-interval', type=int, default=10, help="interval frequency for which the resume state checkpoints are stored.")


    # args = parser.parse_args()
    # if args.num_nearest > args.max_len: args.num_nearest = args.max_len
    # return args


    return parser


def set_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    random.seed(s)

    # keep cudnn stable for CNN，https://learnopencv.com/ensuring-training-reproducibility-in-pytorch/
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger(object):
    level_relations = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING,
                       'error': logging.ERROR, 'crit': logging.CRITICAL}  # 日志级别关系映射 
# Log level relationship mapping
    def __init__(self, path, filename, level='info', rank=0):
        if rank == 0:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
        self.logger = logging.getLogger(str(path / Path(filename)))
        self.logger.setLevel(self.level_relations.get(level))

        sh = logging.StreamHandler()
        self.logger.addHandler(sh)

        th = logging.FileHandler(str(path / Path(filename)), encoding='utf-8')
        self.logger.addHandler(th)


if __name__ == '__main__':
    print()
