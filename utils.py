from __future__ import print_function

import os
import json
import logging
import numpy as np
from tqdm import tqdm, trange
from datetime import datetime
from collections import defaultdict
import _pickle as cPickle

import torch as t
import torch
from torch.autograd import Variable


##########################
# Torch
##########################

def detach(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(detach(v) for v in h)

def get_variable(inputs, cuda=False, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = t.Tensor(inputs)
    if cuda:
        out = Variable(inputs.cuda(), **kwargs)
    else:
        out = Variable(inputs, **kwargs)
    return out

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


##########################
# ETC
##########################

class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret

def get_logger(name=__file__, level=logging.INFO):
    logger = logging.getLogger(name)

    if getattr(logger, '_init_done__', None):
        logger.setLevel(level)
        return logger

    logger._init_done__ = True
    logger.propagate = False
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(0)

    del logger.handlers[:]
    logger.addHandler(handler)

    return logger

logger = get_logger()

def load_pkl(path):
    """
    Load a pickled file.

    :param path: Path to the pickled file.

    :return: The unpickled Python object.
    """
    f = open(path, 'rb')
    try:
        rval = cPickle.load(f)
    finally:
        f.close()
    return rval

def prepare_dirs(args):

    if args.model_name:
        args.model_name = "{}_{}".format(args.dataset, args.model_name)
        if os.path.exists(os.path.join(args.log_dir,args.model_name)):
            raise Exception(f"Model args.model_name already exits !! give a differnt name")
    else:
        if args.load_path:
            args.model_dir = './'+args.log_dir + '/' + args.load_path.split('/')[-2]
        else:
            raise Exception("Atleast one of model name or load path should be specified")

    if not hasattr(args, 'model_dir'):
        args.model_dir = os.path.join(args.log_dir, args.model_name)

    args.data_path = os.path.join(args.data_dir, args.dataset)

    for path in [args.log_dir, args.data_dir, args.model_dir]:
        if not os.path.exists(path):
            makedirs(path)

def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def save_result(vid2pred, vid2GTs, save_fpath):
    assert set(vid2pred.keys()) == set(vid2GTs.keys())

    save_dpath = os.path.dirname(save_fpath)
    if not os.path.exists(save_dpath):
        os.makedirs(save_dpath)

    vids = vid2pred.keys()
    with open(save_fpath, 'w') as fout:
        for vid in vids:
            GTs = ' / '.join(vid2GTs[vid])
            pred = vid2pred[vid]
            # print(GTs)
            # print(pred)
            # print(vid)
            line = ', '.join([str(vid), pred[0], GTs])
            fout.write("{}\n".format(line))


def save_args(args):
    param_path = os.path.join(args.model_dir, "params.json")

    logger.info("[*] MODEL dir: %s" % args.model_dir)
    logger.info("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(args.__dict__, fp, indent=4, sort_keys=True)

def makedirs(path):
    if not os.path.exists(path):
        logger.info("[*] Make directories : {}".format(path))
        os.makedirs(path)

def remove_file(path):
    if os.path.exists(path):
        logger.info("[*] Removed: {}".format(path))
        os.remove(path)

def backup_file(path):
    root, ext = os.path.splitext(path)
    new_path = "{}.backup_{}{}".format(root, get_time(), ext)

    os.rename(path, new_path)
    logger.info("[*] {} has backup: {}".format(path, new_path))

def recnet_local_loss(rec_feats, feats, feats_mask):
    Eds = torch.sqrt(torch.sum(((rec_feats - feats) * feats_mask.unsqueeze(-1)) ** 2, -1))
    return torch.sum(Eds, -1) / torch.sum(feats_mask, -1)

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr