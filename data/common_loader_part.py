#!/usr/bin/env python

# import torch
#
# import data
# import models
# import config
# from utils import *
# from trainer import  Trainer
# from utils import get_logger
import os
import re
import pickle
import nltk
from nltk.tag.stanford import StanfordPOSTagger
import tqdm
import argparse
import json
import os
import re
import sys

from allennlp.predictors.predictor import Predictor

from nltk.tokenize import TreebankWordTokenizer

MODELS_DIR = '/data1/tuyunbin/RGAT-ABSA/'
model_path = os.path.join(
    MODELS_DIR, "biaffine-dependency-parser-ptb-2018.08.23.tar.gz")
os.environ['CUDA_VISIBLE_DEVICES'] = '2'



def main():
    st = StanfordPOSTagger(
        "/data1/tuyunbin/PR/data/stanford-tagger-4.2.0/stanford-postagger-full-2020-11-17/models/english-bidirectional-distsim.tagger",
        "/data1/tuyunbin/PR/data/stanford-tagger-4.2.0/stanford-postagger-full-2020-11-17/stanford-postagger.jar", encoding='utf8')
    # prepare_dirs(args)
    _captions_path = "/data1/tuyunbin/PR/data/msrvtt16/CAP.pkl"
    predictor = Predictor.from_path(model_path)
    with open(_captions_path, 'rb') as f:
        caption = pickle.load(f, encoding='iso-8859-1')
        print('Predicting dependency information...')
        for kk, vv in tqdm.tqdm(caption.items(), desc=" POS tagging =>", total=len(caption)):
            for i in vv:
                cap = i['tokenized']
                r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，]+'
                cap = re.sub(r, '', cap)
                lengths = len(cap.split())
                doc = predictor.predict(sentence=cap)
                seq_pos = doc['pos'][:lengths]
                plenghts = len(seq_pos)
                if lengths != plenghts:
                    print(kk)
                i['pos'] = seq_pos

    with open("/data1/tuyunbin/PR/data/msrvtt16/CAP_with_POS.pkl", 'wb') as fo:  # 将数据写入pkl文件
        pickle.dump(caption, fo)


if __name__ == "__main__":
    # args, unparsed = config.get_args()
    main()
