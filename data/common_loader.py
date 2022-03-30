from __future__ import print_function
import json
import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import re
import utils

import h5py
np.random.seed(111)

#sys.path.append('/playpen1/home/ram/video_caption_eval')
#from automatic_evaluation import evaluate

PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[END]' # This has a vocab id, which is used at the end of untruncated target sequences


class MSVD_Vocab(object):
    def __init__(self, vocab_file, max_size, vg_path, pos_path):
        """Creates a vocab of up to max_size words, reading from the vocab_file. If max_size is 0, reads the entire vocab file.
            Args:
                vocab_file: path to the vocab file, which is assumed to contain "<word> <frequency>" on each line, sorted with most frequent word first.
                            This code doesn't actually use the frequencies, though.
                max_size: integer. The maximum size of the resulting Vocabulary.

        """
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0  # keeps track of total number of words in the Vocab

        # [PAD], [START], [STOP] and [UNK] get the ids 0,1,2,3.
        for w in [PAD_TOKEN, START_DECODING, STOP_DECODING, UNKNOWN_TOKEN]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'rb') as vocab_f:

            worddict = pickle.load(vocab_f,encoding='iso-8859-1')
            for kk, vv in worddict.items():

                w = kk
                if w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception('[UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self._word_to_id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (
                    max_size, self._count))
                    break

        with open(vg_path, 'r') as f:
            line = f.readline()
            line = line[:-1]
            semantic_object = []
            while line:
                semantic_object.append(line)
                line = f.readline()
                line = line[:-1]


        new_s_os = []
        for o in semantic_object:
            new_s_o = o.split('_')
            for i in new_s_o:
                if i not in new_s_os:
                    new_s_os.append(i)
        new_s_os_l = len(new_s_os)

        for oo in new_s_os:

            # if oo not in self.word_idict.values():
            if oo in self._word_to_id:
                continue
            else:
                self._word_to_id[oo] = self._count

            self._id_to_word[self._count] = oo

            self._count += 1
            if max_size != 0 and self._count >= max_size:
                print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (
                max_size, self._count))
                break

        with open(pos_path, 'rb') as vocab_f:

            pos = pickle.load(vocab_f,encoding='iso-8859-1')
            self._pos_to_id = pos['pos_to_id']
            self._id_to_pos = pos['id_to_pos']

        print("Finished constructing vocabulary of %i total words. Last word added: %s" % (
        self._count, self._id_to_word[self._count - 1]))


    def pos2id(self, pos):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        if pos not in self._pos_to_id:
            return self._pos_to_id[UNKNOWN_TOKEN]
        return self._pos_to_id[pos]

    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            # raise ValueError('Id not found in vocab: %d' % word_id)
            word_id = 0
        return self._id_to_word[word_id]

    def print_id2word(self):
        print(self._id_to_word)

    def size(self):
        """Returns the total size of the vocabulary"""
        return self._count

    def texttoidx(self, text, max_sentence_length, add_start_token=False):
        text = text + ' ' + STOP_DECODING
        if add_start_token:
            text = START_DECODING + ' ' + text
        tokens = []
        seq_length = 0
        for word in text.split()[:max_sentence_length]:  # also need one more for [END] token
            tokens.append(self.word2id(word))
            seq_length += 1

        tokens.extend([0 for i in range(max_sentence_length - len(tokens))])

        return np.asarray(tokens), seq_length

    def postoidx(self, pos, max_sentence_length, add_start_token=False):
        # pos = " ".join(pos_list)
        pos.append(STOP_DECODING)
        if add_start_token:
            pos.insert(0, START_DECODING)
        tokens = []
        seq_length = 0
        for p in pos[:max_sentence_length]:
            # also need one more for [END] token
            tokens.append(self.pos2id(p))
            seq_length += 1

        tokens.extend([0 for i in range(max_sentence_length - len(tokens))])

        return np.asarray(tokens), seq_length

    def tagtoidx(self, tag, max_sentence_length, add_start_token=False):
        # text = text + ' ' + STOP_DECODING
        # if add_start_token:
        #     text = START_DECODING + ' ' + text
        tags = []
        seq_length = 0
        for ta in tag[:max_sentence_length]:  # also need one more for [END] token
            tags.append(self.word2id(ta))
            seq_length += 1

        tags.extend([0 for i in range(max_sentence_length - len(tags))])
        if tags == []:
           tags.append(self.word2id('[PAD]'))
           seq_length = 1

        return np.asarray(tags), seq_length


class Vocab(object):
    def __init__(self, vocab_file, max_size, vg_path, pos_path):
        """Creates a vocab of up to max_size words, reading from the vocab_file. If max_size is 0, reads the entire vocab file.
            Args:
                vocab_file: path to the vocab file, which is assumed to contain "<word> <frequency>" on each line, sorted with most frequent word first.
                            This code doesn't actually use the frequencies, though.
                max_size: integer. The maximum size of the resulting Vocabulary.

        """
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0  # keeps track of total number of words in the Vocab

        # [PAD], [START], [STOP] and [UNK] get the ids 0,1,2,3.
        for w in [PAD_TOKEN, START_DECODING, STOP_DECODING, UNKNOWN_TOKEN]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r') as vocab_f:
            for line in vocab_f.read().splitlines():
                pieces = line.split('\t')
                if len(pieces) != 2:
                    print('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
                    continue
                w = pieces[0]
                if w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception('[UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self._word_to_id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (
                    max_size, self._count))
                    break

        with open(vg_path, 'r') as f:
            line = f.readline()
            line = line[:-1]
            semantic_object = []
            while line:
                semantic_object.append(line)
                line = f.readline()
                line = line[:-1]


        new_s_os = []
        for o in semantic_object:
            new_s_o = o.split('_')
            for i in new_s_o:
                if i not in new_s_os:
                    new_s_os.append(i)
        new_s_os_l = len(new_s_os)

        for oo in new_s_os:

            # if oo not in self.word_idict.values():
            if oo in self._word_to_id:
                continue
            else:
                self._word_to_id[oo] = self._count

            self._id_to_word[self._count] = oo

            self._count += 1
            if max_size != 0 and self._count >= max_size:
                print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (
                max_size, self._count))
                break

        with open(pos_path, 'rb') as vocab_f:

            pos = pickle.load(vocab_f,encoding='iso-8859-1')
            self._pos_to_id = pos['pos_to_id']
            self._id_to_pos = pos['id_to_pos']

        print("Finished constructing vocabulary of %i total words. Last word added: %s" % (
        self._count, self._id_to_word[self._count - 1]))


    def pos2id(self, pos):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        if pos not in self._pos_to_id:
            return self._pos_to_id[UNKNOWN_TOKEN]
        return self._pos_to_id[pos]

    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            word_id = 0
            # raise ValueError('Id not found in vocab: %d' % word_id)

        return self._id_to_word[word_id]

    def print_id2word(self):
        print(self._id_to_word)

    def size(self):
        """Returns the total size of the vocabulary"""
        return self._count

    def texttoidx(self, text, max_sentence_length, add_start_token=False):
        text = text + ' ' + STOP_DECODING
        if add_start_token:
            text = START_DECODING + ' ' + text
        tokens = []
        seq_length = 0
        for word in text.split()[:max_sentence_length]:  # also need one more for [END] token
            tokens.append(self.word2id(word))
            seq_length += 1

        tokens.extend([0 for i in range(max_sentence_length - len(tokens))])

        return np.asarray(tokens), seq_length

    def postoidx(self, pos, max_sentence_length, add_start_token=False):
        # pos = " ".join(pos_list)
        pos.append(STOP_DECODING)
        if add_start_token:
            pos.insert(0, START_DECODING)
        tokens = []
        seq_length = 0
        for p in pos[:max_sentence_length]:
            # also need one more for [END] token
            tokens.append(self.pos2id(p))
            seq_length += 1

        tokens.extend([0 for i in range(max_sentence_length - len(tokens))])

        return np.asarray(tokens), seq_length

    def tagtoidx(self, tag, max_sentence_length, add_start_token=False):
        # text = text + ' ' + STOP_DECODING
        # if add_start_token:
        #     text = START_DECODING + ' ' + text
        tags = []
        seq_length = 0
        for ta in tag[:max_sentence_length]:  # also need one more for [END] token
            tags.append(self.word2id(ta))
            seq_length += 1

        tags.extend([0 for i in range(max_sentence_length - len(tags))])
        if tags == []:
           tags.append(self.word2id('[PAD]'))
           seq_length = 1

        return np.asarray(tags), seq_length

class Vocab_old(object):
    def __init__(self, vocab_file,max_size, vg_path):
        """Creates a vocab of up to max_size words, reading from the vocab_file. If max_size is 0, reads the entire vocab file.
            Args:
                vocab_file: path to the vocab file, which is assumed to contain "<word> <frequency>" on each line, sorted with most frequent word first. 
                            This code doesn't actually use the frequencies, though.
                max_size: integer. The maximum size of the resulting Vocabulary.
                
        """
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0 # keeps track of total number of words in the Vocab

        # [PAD], [START], [STOP] and [UNK] get the ids 0,1,2,3.
        for w in [PAD_TOKEN, START_DECODING, STOP_DECODING, UNKNOWN_TOKEN]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r') as vocab_f:
            for line in vocab_f.read().splitlines():
                pieces = line.split('\t')
                if len(pieces) != 2:
                    print('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
                    continue
                w = pieces[0]
                if w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception('[UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self._word_to_id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count))
                    break

        with open(vg_path, 'r') as f:
            line = f.readline()
            line = line[:-1]
            semantic_object = []
            while line:
                semantic_object.append(line)
                line = f.readline()
                line = line[:-1]


        new_s_os = []
        for o in semantic_object:
            new_s_o = o.split('_')
            for i in new_s_o:
                if i not in new_s_os:
                    new_s_os.append(i)
        new_s_os_l = len(new_s_os)

        for oo in new_s_os:

            # if oo not in self.word_idict.values():
            if oo in self._word_to_id:
                continue
            else:
                self._word_to_id[oo] = self._count

            self._id_to_word[self._count] = oo

            self._count += 1
            if max_size != 0 and self._count >= max_size:
                print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (
                max_size, self._count))
                break

        print("Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id_to_word[self._count-1]))

    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def tagtoidx(self, tag, max_sentence_length, add_start_token=False):
        # text = text + ' ' + STOP_DECODING
        # if add_start_token:
        #     text = START_DECODING + ' ' + text
        tags = []
        seq_length = 0
        for ta in tag[:max_sentence_length]:  # also need one more for [END] token
            tags.append(self.word2id(ta))
            seq_length += 1

        tags.extend([0 for i in range(max_sentence_length - len(tags))])
        if tags == []:
           tags.append(self.word2id('[PAD]'))
           seq_length = 1

        return np.asarray(tags), seq_length

    def print_id2word(self):
        print(self._id_to_word)

    def size(self):
        """Returns the total size of the vocabulary"""
        return self._count

    def texttoidx(self,text,max_sentence_length, add_start_token=False):
        text = text + ' ' + STOP_DECODING
        if add_start_token:
            text = START_DECODING + ' ' + text
        tokens = []
        seq_length = 0
        for word in text.split()[:max_sentence_length]: # also need one more for [END] token
            tokens.append(self.word2id(word))
            seq_length += 1

        tokens.extend([0 for i in range(max_sentence_length-len(tokens))])

        return np.asarray(tokens),seq_length


class Batch(object):

    def __init__(self):
        self._dict = {}


    def put(self,key,value):
        if self._dict.get(key) is None:
            self._dict[key] = value
        else:
            raise Exception("key:{} already exits".format(key))

    def get(self,key):
       if self._dict.get(key) is not None:
           return self._dict[key]
       else:
           raise Exception("key:{} already exits".format(key))


class MSVDBatcher(object):

    def __init__(self, hps, mode, vocab):

        self._vid_feature_path = hps.msvd_vid_feature_path
        self.tag_path = hps.msvd_tag_path
        self._captions_path = hps.msvd_captions_path
        self._max_enc_steps = hps.encoder_rnn_max_length
        self._max_dec_steps = hps.decoder_rnn_max_length
        self._max_tag_steps = hps.tag_max_length
        self._mode = mode
        self._batch_size = hps.batch_size
        self.vocab = vocab
        self.vg_path = hps.msvd_vg_path
        self._vid_dim = hps.vid_dim
        self.train_ids = ['vid%s' % i for i in range(1, 1201)]
        self.valid_ids = ['vid%s' % i for i in range(1201, 1301)]
        self.test_ids = ['vid%s' % i for i in range(1301, 1971)]
        self.data, self.data_dict, self.pos_dict = self._process_data()
        self.num_steps = int(len(self.data) / self._batch_size) + 1

    def _process_data(self):
        """this module extracts data from videos and caption files and creates batches"""
        # load json data which contains all the information
        data = []
        data_dict = {}
        pos_dict = {}

        # filename = 'sents_' + self._mode + '.txt'
        if self._mode == 'train':
            with open("data/msvd/train.pkl", 'rb') as fin:
                IDs = pickle.load(fin, encoding='iso-8859-1')
        elif self._mode == 'val':
            with open("data/msvd/valid.pkl", 'rb') as fin:
                IDs = pickle.load(fin, encoding='iso-8859-1')
        elif self._mode == 'test':
            with open("data/msvd/test.pkl", 'rb') as fin:
                IDs = pickle.load(fin, encoding='iso-8859-1')

        with open(self._captions_path, 'rb') as f:
            caption = pickle.load(f,encoding='iso-8859-1')
            for i, ID in enumerate(IDs):
                vidID, capID = ID.split('_')
                caps = caption[vidID]
                rval = None
                for cap in caps:
                    if cap['cap_id'] == capID:
                        rval = cap['tokenized'].lower()
                        r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，]+'
                        rval = re.sub(r, '', rval)
                        pos = cap['pos']

                        data.append((vidID, rval, pos))
                        if data_dict.get(vidID) is None:
                            data_dict[vidID] = [rval]
                        else:
                            data_dict[vidID].append(rval)

                        if pos_dict.get(vidID) is None:
                            pos_dict[vidID] = [pos]
                        else:
                            pos_dict[vidID].append(pos)
                        break
                assert rval is not None


            # for kk, vv in caption.items():
            #
            #     vid_id = kk
            #     caption = vv
            #     data.append((vid_id, caption))
            #     if data_dict.get(vid_id) is None:
            #         data_dict[vid_id] = [caption]
            #     else:
            #         data_dict[vid_id].append(caption)

        if self._mode == 'train':
            np.random.shuffle(data)
        else:
            data, _, pos = zip(*data)  # consider only video ids for evaluation
            data = sorted(set(data), key=data.index)

        return data, data_dict, pos_dict

    def sort_based_on_caption_lengths(self, video_batch, video_len_batch, video_id, caption_batch, caption_len_batch,
                                      original_caption, tags_batch, tags_len_batch, pos_batch, pos_len_batch, original_pos):
        sorted_indices = np.argsort(caption_len_batch)[::-1]
        return video_batch[sorted_indices], video_len_batch[sorted_indices], video_id[sorted_indices], caption_batch[
            sorted_indices], caption_len_batch[sorted_indices], original_caption[sorted_indices], tags_batch[sorted_indices], tags_len_batch[sorted_indices],\
               pos_batch[sorted_indices], pos_len_batch[sorted_indices], original_pos[sorted_indices]

    def get_batcher(self):
        """
        This module process data and creates batches for train/val/test
        Also acts as generator
        """
        if self._mode == 'train':
            np.random.shuffle(self.data)

        for i in range(0, len(self.data), self._batch_size):
            start = i
            if i + self._batch_size > len(self.data):  # handling leftovers
                end = len(self.data)
                current_batch_size = end - start
            else:
                end = i + self._batch_size
                current_batch_size = self._batch_size
            if self._mode == 'train':
                video_id, original_caption, original_pos = zip(*self.data[start:end])
            else:
                video_id = self.data[start:end]

            fin = h5py.File(self._vid_feature_path, 'r')
            tag_file = utils.load_pkl(self.tag_path)
            tag = [tag_file[vid] for vid in video_id]

            video_features = [fin[vid][()] for vid in video_id]
            # fin = h5py.File(fpath, 'r')
            # for vid in fin.keys():
            #     feats = fin[vid].value
            #
            # video_features = [np.load(os.path.join(self._vid_feature_path,key+'.mp4.npy')) for key in video_id]

            if self._mode == 'train':
                caption_batch = []
                caption_length = []
                pos_batch = []
                pos_length = []
                for cap in original_caption:
                    cap_id, cap_length = self.vocab.texttoidx(cap, self._max_dec_steps)
                    caption_batch.append(cap_id)
                    caption_length.append(cap_length)
                for pos in original_pos:
                    pos_id, p_length = self.vocab.postoidx(pos, self._max_dec_steps)
                    pos_batch.append(pos_id)
                    pos_length.append(p_length)

            original_caption_dict = {}
            for vid in video_id:
                original_caption_dict[vid] = self.data_dict[vid]

            original_pos_dict = {}
            for vid in video_id:
                original_pos_dict[vid] = self.pos_dict[vid]

            tags_batch = []
            tags_length = []
            for ta in tag:
                if ta == []:
                    ta = ['[PAD]']
                tag_id, tag_length = self.vocab.tagtoidx(ta, self._max_tag_steps)
                tags_batch.append(tag_id)
                tags_length.append(tag_length)

            # transform/clip frames
            video_batch = np.zeros((current_batch_size, self._max_enc_steps, self._vid_dim))
            video_length = []
            for idx, feat in enumerate(video_features):
                if len(feat) > self._max_enc_steps:
                    video_batch[idx][:] = feat[:self._max_enc_steps]
                    video_length.append(self._max_enc_steps)
                else:
                    video_batch[idx][:len(feat)] = feat
                    video_length.append(len(feat))

            if self._mode == 'train':
                video_batch, video_length, video_id, caption_batch, caption_length, original_caption, tags_batch, tags_length, pos_batch, pos_length, original_pos = self.sort_based_on_caption_lengths(
                    np.asarray(video_batch), np.asarray(video_length),
                    np.asarray(video_id), np.asarray(caption_batch),
                    np.asarray(caption_length), np.asarray(original_caption),
                    np.asarray(tags_batch), np.asarray(tags_length),
                    np.asarray(pos_batch), np.asarray(pos_length), np.asarray(original_pos))

            else:
                video_batch = np.asarray(video_batch)
                video_length = np.asarray(video_length)
                tags_batch = np.asarray(tags_batch)
                tags_length = np.asarray(tags_length)

            batch = Batch()
            if self._mode == 'train':
                batch.put('original_caption', original_caption)
                batch.put('caption_batch', torch.LongTensor(caption_batch))
                batch.put('caption_len_batch', caption_length)
                batch.put('original_pos', original_pos)
                batch.put('pos_batch', torch.LongTensor(pos_batch))
                batch.put('pos_len_batch', pos_length)
            batch.put('original_caption_dict', original_caption_dict)
            batch.put('original_pos_dict', original_pos_dict)
            batch.put('video_batch', torch.FloatTensor(video_batch))
            batch.put('video_len_batch', video_length)
            batch.put('video_id', video_id)
            batch.put('tag_batch', torch.LongTensor(tags_batch))
            batch.put('tag_len_batch', tags_length)
            yield batch


class MSRVTTBatcher(object):

    def __init__(self, hps, mode, vocab):

        self._vid_feature_path = hps.vid_feature_path
        self.tag_path = hps.tag_path
        self._captions_path = hps.captions_path
        self._max_enc_steps = hps.encoder_rnn_max_length
        self._max_dec_steps = hps.decoder_rnn_max_length
        self._max_tag_steps = hps.tag_max_length
        self._mode = mode
        self._batch_size = hps.batch_size
        self.vocab = vocab
        self.vg_path = hps.msvd_vg_path
        self._vid_dim = hps.vid_dim
        self.train_ids = ['video%s' % i for i in range(0, 6513)]
        self.valid_ids = ['video%s' % i for i in range(6513, 7010)]
        self.test_ids = ['video%s' % i for i in range(7010, 10000)]
        self.data, self.data_dict, self.pos_dict = self._process_data()
        self.num_steps = int(len(self.data) / self._batch_size) + 1

    def _process_data(self):
        """this module extracts data from videos and caption files and creates batches"""
        # load json data which contains all the information
        data = []
        data_dict = {}
        pos_dict = {}

        # filename = 'sents_' + self._mode + '.txt'
        if self._mode == 'train':
            with open("data/msrvtt16/train.pkl", 'rb') as fin:
                IDs = pickle.load(fin, encoding='iso-8859-1')
        elif self._mode == 'val':
            with open("data/msrvtt16/valid.pkl", 'rb') as fin:
                IDs = pickle.load(fin, encoding='iso-8859-1')
        elif self._mode == 'test':
            with open("data/msrvtt16/test.pkl", 'rb') as fin:
                IDs = pickle.load(fin, encoding='iso-8859-1')

        with open(self._captions_path, 'rb') as f:
            caption = pickle.load(f,encoding='iso-8859-1')
            for i, ID in enumerate(IDs):
                vidID, capID = ID.split('_')
                caps = caption[vidID]
                rval = None
                for cap in caps:
                    if cap['cap_id'] == capID:
                        rval = cap['tokenized'].lower()
                        r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，]+'
                        rval = re.sub(r, '', rval)
                        pos = cap['pos']

                        data.append((vidID, rval, pos))
                        if data_dict.get(vidID) is None:
                            data_dict[vidID] = [rval]
                        else:
                            data_dict[vidID].append(rval)

                        if pos_dict.get(vidID) is None:
                            pos_dict[vidID] = [pos]
                        else:
                            pos_dict[vidID].append(pos)
                        break
                assert rval is not None


            # for kk, vv in caption.items():
            #
            #     vid_id = kk
            #     caption = vv
            #     data.append((vid_id, caption))
            #     if data_dict.get(vid_id) is None:
            #         data_dict[vid_id] = [caption]
            #     else:
            #         data_dict[vid_id].append(caption)

        if self._mode == 'train':
            np.random.shuffle(data)
        else:
            data, _, pos = zip(*data)  # consider only video ids for evaluation
            data = sorted(set(data), key=data.index)

        return data, data_dict, pos_dict

    def sort_based_on_caption_lengths(self, video_batch, video_len_batch, video_id, caption_batch, caption_len_batch,
                                      original_caption, tags_batch, tags_len_batch, pos_batch, pos_len_batch, original_pos):
        sorted_indices = np.argsort(caption_len_batch)[::-1]
        return video_batch[sorted_indices], video_len_batch[sorted_indices], video_id[sorted_indices], caption_batch[
            sorted_indices], caption_len_batch[sorted_indices], original_caption[sorted_indices], tags_batch[sorted_indices], tags_len_batch[sorted_indices],\
               pos_batch[sorted_indices], pos_len_batch[sorted_indices], original_pos[sorted_indices]

    def get_batcher(self):
        """
        This module process data and creates batches for train/val/test
        Also acts as generator
        """
        if self._mode == 'train':
            np.random.shuffle(self.data)

        for i in range(0, len(self.data), self._batch_size):
            start = i
            if i + self._batch_size > len(self.data):  # handling leftovers
                end = len(self.data)
                current_batch_size = end - start
            else:
                end = i + self._batch_size
                current_batch_size = self._batch_size
            if self._mode == 'train':
                video_id, original_caption, original_pos = zip(*self.data[start:end])
            else:
                video_id = self.data[start:end]

            fin = h5py.File(self._vid_feature_path, 'r')
            tag_file = utils.load_pkl(self.tag_path)
            tag = [tag_file[vid] for vid in video_id]

            video_features = [fin[vid][()] for vid in video_id]
            # fin = h5py.File(fpath, 'r')
            # for vid in fin.keys():
            #     feats = fin[vid].value
            #
            # video_features = [np.load(os.path.join(self._vid_feature_path,key+'.mp4.npy')) for key in video_id]

            if self._mode == 'train':
                caption_batch = []
                caption_length = []
                pos_batch = []
                pos_length = []
                for cap in original_caption:
                    cap_id, cap_length = self.vocab.texttoidx(cap, self._max_dec_steps)
                    caption_batch.append(cap_id)
                    caption_length.append(cap_length)
                for pos in original_pos:
                    pos_id, p_length = self.vocab.postoidx(pos, self._max_dec_steps)
                    pos_batch.append(pos_id)
                    pos_length.append(p_length)

            original_caption_dict = {}
            for vid in video_id:
                original_caption_dict[vid] = self.data_dict[vid]

            original_pos_dict = {}
            for vid in video_id:
                original_pos_dict[vid] = self.pos_dict[vid]

            tags_batch = []
            tags_length = []
            for ta in tag:
                if ta == []:
                    ta = ['[PAD]']
                tag_id, tag_length = self.vocab.tagtoidx(ta, self._max_tag_steps)
                tags_batch.append(tag_id)
                tags_length.append(tag_length)

            # transform/clip frames
            video_batch = np.zeros((current_batch_size, self._max_enc_steps, self._vid_dim))
            video_length = []
            for idx, feat in enumerate(video_features):
                if len(feat) > self._max_enc_steps:
                    video_batch[idx][:] = feat[:self._max_enc_steps]
                    video_length.append(self._max_enc_steps)
                else:
                    video_batch[idx][:len(feat)] = feat
                    video_length.append(len(feat))

            if self._mode == 'train':
                video_batch, video_length, video_id, caption_batch, caption_length, original_caption, tags_batch, tags_length, pos_batch, pos_length, original_pos = self.sort_based_on_caption_lengths(
                    np.asarray(video_batch), np.asarray(video_length),
                    np.asarray(video_id), np.asarray(caption_batch),
                    np.asarray(caption_length), np.asarray(original_caption),
                    np.asarray(tags_batch), np.asarray(tags_length),
                    np.asarray(pos_batch), np.asarray(pos_length), np.asarray(original_pos))

            else:
                video_batch = np.asarray(video_batch)
                video_length = np.asarray(video_length)
                tags_batch = np.asarray(tags_batch)
                tags_length = np.asarray(tags_length)

            batch = Batch()
            if self._mode == 'train':
                batch.put('original_caption', original_caption)
                batch.put('caption_batch', torch.LongTensor(caption_batch))
                batch.put('caption_len_batch', caption_length)
                batch.put('original_pos', original_pos)
                batch.put('pos_batch', torch.LongTensor(pos_batch))
                batch.put('pos_len_batch', pos_length)
            batch.put('original_caption_dict', original_caption_dict)
            batch.put('original_pos_dict', original_pos_dict)
            batch.put('video_batch', torch.FloatTensor(video_batch))
            batch.put('video_len_batch', video_length)
            batch.put('video_id', video_id)
            batch.put('tag_batch', torch.LongTensor(tags_batch))
            batch.put('tag_len_batch', tags_length)
            yield batch


class MSRVTTBatcher_old(object):

    def __init__(self,hps,mode,vocab):
        
        self._vid_feature_path = hps.vid_feature_path
        self._captions_path = hps.captions_path
        self.tag_path = hps.tag_path
        self._max_enc_steps = hps.encoder_rnn_max_length
        self._max_dec_steps = hps.decoder_rnn_max_length
        self._mode = mode
        self._batch_size = hps.batch_size
        self.vocab = vocab
        self._vid_dim = hps.vid_dim
        self.data,self.data_dict = self._process_data()
        self.num_steps = int(len(self.data)/self._batch_size) + 1

    def _process_data(self):
        """this module extracts data from videos and caption files and creates batches"""
        # load json data which contains all the information
        data = []
        data_dict = {}
        filename ='sents_'+self._mode+'.txt'
        with open(os.path.join(self._captions_path,filename),'r') as f:
            for line in f.read().splitlines():
                line = line.split('\t')
                vid_id = line[0]
                caption = line[1]
                data.append((vid_id,caption))
                if data_dict.get(vid_id) is None:
                    data_dict[vid_id] = [caption]
                else:
                    data_dict[vid_id].append(caption)
      
        if self._mode == 'train':
            np.random.shuffle(data)
        else:
            data,_ = zip(*data) # consider only video ids for evaluation
            data = sorted(set(data),key=data.index)

        return data,data_dict

    def sort_based_on_caption_lengths(self, video_batch, video_len_batch, video_id, caption_batch, caption_len_batch,
                                      original_caption, tags_batch, tags_len_batch):
        sorted_indices = np.argsort(caption_len_batch)[::-1]
        return video_batch[sorted_indices], video_len_batch[sorted_indices], video_id[sorted_indices], caption_batch[
            sorted_indices], caption_len_batch[sorted_indices], original_caption[sorted_indices], tags_batch[
                   sorted_indices], tags_len_batch[sorted_indices]
    
    def get_batcher(self):
        """
        This module process data and creates batches for train/val/test 
        Also acts as generator
        """
        if self._mode == 'train':
            np.random.shuffle(self.data)
        
        for i in range(0,len(self.data),self._batch_size):
            start = i
            if i+self._batch_size > len(self.data): # handling leftovers
                end = len(self.data)
                current_batch_size = end-start
            else:
                end = i+self._batch_size
                current_batch_size = self._batch_size
            if self._mode == 'train':
                video_id,original_caption = zip(*self.data[start:end])
            else:
                video_id = self.data[start:end]


            fin = h5py.File(self._vid_feature_path, 'r')
            tag_file = utils.load_pkl(self.tag_path)
            tag = [tag_file[vid] for vid in video_id]

            video_features = [fin[vid][()] for vid in video_id]
            # fin = h5py.File(fpath, 'r')
            # for vid in fin.keys():
            #     feats = fin[vid].value
            #
            # video_features = [np.load(os.path.join(self._vid_feature_path,key+'.mp4.npy')) for key in video_id]
            
            if self._mode == 'train':
                caption_batch = []
                caption_length = []
                for cap in original_caption:
                    cap_id,cap_length = self.vocab.texttoidx(cap,self._max_dec_steps)
                    caption_batch.append(cap_id)
                    caption_length.append(cap_length)

            original_caption_dict = {}
            for vid in video_id:
                original_caption_dict[vid] = self.data_dict[vid]

            tags_batch = []
            tags_length = []
            for ta in tag:
                if ta == []:
                    ta = ['[PAD]']
                tag_id, tag_length = self.vocab.tagtoidx(ta, 20)
                tags_batch.append(tag_id)
                tags_length.append(tag_length)
                
            # transform/clip frames
            video_batch = np.zeros((current_batch_size,self._max_enc_steps,self._vid_dim))
            video_length = []
            for idx,feat in enumerate(video_features):
                if len(feat)>self._max_enc_steps:
                    video_batch[idx][:] = feat[:self._max_enc_steps]
                    video_length.append(self._max_enc_steps)
                else:
                    video_batch[idx][:len(feat)] = feat
                    video_length.append(len(feat))

            if self._mode == 'train':
                video_batch, video_length, video_id, caption_batch, caption_length, original_caption, tags_batch, tags_length = self.sort_based_on_caption_lengths(
                    np.asarray(video_batch), np.asarray(video_length),
                    np.asarray(video_id), np.asarray(caption_batch),
                    np.asarray(caption_length), np.asarray(original_caption),
                    np.asarray(tags_batch), np.asarray(tags_length))

            else:
                video_batch = np.asarray(video_batch)
                video_length = np.asarray(video_length)
                tags_batch = np.asarray(tags_batch)
                tags_length = np.asarray(tags_length)

            batch = Batch()
            if self._mode == 'train':
                batch.put('original_caption', original_caption)
                batch.put('caption_batch', torch.LongTensor(caption_batch))
                batch.put('caption_len_batch', caption_length)
            batch.put('original_caption_dict', original_caption_dict)
            batch.put('video_batch', torch.FloatTensor(video_batch))
            batch.put('video_len_batch', video_length)
            batch.put('video_id', video_id)
            batch.put('tag_batch', torch.LongTensor(tags_batch))
            batch.put('tag_len_batch', tags_length)
            yield batch




class SNLIBatcher(object):

    def __init__(self,max_steps,vocab):
        self._max_steps = max_steps
        self.vocab = vocab


    def process_external_data(self, prem, hypo):

        original_premise = prem
        original_hypothesis = hypo


        premise_batch = []
        premise_length = []
        hypothesis_batch = []
        hypothesis_length = []
        
        for prem, hypo in zip(original_premise, original_hypothesis):

            prem_id, prem_length = self.vocab.texttoidx(prem, self._max_steps, add_start_token=True)
            hypo_id, hypo_length = self.vocab.texttoidx(hypo, self._max_steps, add_start_token=True)
            premise_batch.append(prem_id)
            premise_length.append(prem_length)
            hypothesis_batch.append(hypo_id)
            hypothesis_length.append(hypo_length)



        batch = Batch()
        batch.put('original_premise', original_premise)
        batch.put('original_hypothesis', original_hypothesis)
        batch.put('premise_batch', torch.LongTensor(np.asarray(premise_batch)))
        batch.put('premise_length', np.asarray(premise_length))
        batch.put('hypothesis_batch', torch.LongTensor(np.asarray(hypothesis_batch)))
        batch.put('hypothesis_length', np.asarray(hypothesis_length))

        return batch

