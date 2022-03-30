#!/usr/bin/env python

import torch

import data
import models
import config
from utils import *
from trainer import  Trainer
from utils import get_logger
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

logger = get_logger()


def main(args):
    prepare_dirs(args)

    torch.manual_seed(args.random_seed)

    if args.num_gpu > 0:
        torch.cuda.manual_seed(args.random_seed)


    if args.network_type == 'seq2seq' or 'TTA' or 'AVS' or 'TTA_AVS':
        if args.dataset == 'msrvtt':
            vocab = data.common_loader.Vocab(args.vocab_file, args.max_vocab_size, args.vg_path, args.pos_file)
        elif args.dataset =='msvd':
            vocab = data.common_loader.MSVD_Vocab(args.msvd_vocab_file, args.msvd_max_vocab_size, args.msvd_vg_path, args.msvd_pos_file)
        dataset = {}
        if args.dataset == 'msrvtt':
            dataset['train'] = data.common_loader.MSRVTTBatcher(args, 'train', vocab)
            dataset['val'] = data.common_loader.MSRVTTBatcher(args, 'val', vocab)
            dataset['test'] = data.common_loader.MSRVTTBatcher(args, 'test', vocab)
        elif args.dataset == 'msvd':
            dataset['train'] = data.common_loader.MSVDBatcher(args, 'train', vocab)
            dataset['val'] = data.common_loader.MSVDBatcher(args, 'val', vocab)
            dataset['test'] = data.common_loader.MSVDBatcher(args, 'test', vocab)

        else:
            raise Exception(f"Unknown dataset: {args.dataset} for the corresponding network type: {args.network_type}")

    else:
        raise NotImplemented(f"{args.dataset} is not supported")

    trainer = Trainer(args, dataset)

    if args.mode == 'train':
        save_args(args)
        trainer.train()
    else:
        if not args.load_path:
            raise Exception("[!] You should specify `load_path` to load a pretrained model")
        else:
            trainer.test(args.mode)

if __name__ == "__main__":
    args, unparsed = config.get_args()
    main(args)
