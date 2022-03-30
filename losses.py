import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def entropy_loss(x, ignore_mask):
    b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    b = b.sum(dim=2)
    b[ignore_mask] = 0 # Mask after sum to avoid memory issue.
    b = -1.0 * b.sum(dim=0).mean() # Sum along words and mean along batch
    return b


def refine_loss(x, x_refine, keep_mask):
    # x = x.mean(dim=1)

    # caption_len = keep_mask.sum(dim=1) #(B)
    # caption_len = caption_len.unsqueeze(1).expand(caption_len.size(0), x_refine.size(2))#(B, dim)
    # caption_len = caption_len.type(torch.cuda.FloatTensor)
    keep_mask = keep_mask.unsqueeze(2).expand_as(x_refine).type(torch.cuda.FloatTensor)#(B, T,dim)
    x_refine = keep_mask * x_refine
    x = keep_mask * x
    l2_loss = torch.sum((x-x_refine)**2, -1).mean(1).mean()

    # x_refine = x_refine.sum(dim=1) / caption_len
    #
    # l2_loss = F.mse_loss(x, x_refine)
    return l2_loss

def ac_loss(att_weights):
    att_weights = att_weights.transpose(0,1)
    len_a = att_weights.size(2)
    last = att_weights[:,:,1:len_a]
    pre = att_weights[:,:,:len_a-1]
    # l2_loss = F.mse_loss(last, pre)
    l2_loss = (last-pre).mean()
    return l2_loss


def relevance_loss(x, x_recon):
    l2_loss = F.mse_loss(x, x_recon)
    return l2_loss

def coherence_loss(x, x_pre, caption_mask):
    # len_feat = x.size(1)
    # x_last = x[:,:len_feat-1,:]
    # caption_mask = caption_mask.unsqueeze(2).expand_as(x).type(torch.cuda.FloatTensor)
    # caption_mask = caption_mask[:,:len_feat-1,:]
    # x_pre = x_pre * caption_mask
    # x_last = x_last * caption_mask
    #
    # l2_loss = F.mse_loss(x_last, x_pre)
    # len_feat = x.size(1)

    caption_mask = caption_mask.unsqueeze(2).expand_as(x).type(torch.cuda.FloatTensor)

    x_pre = x_pre * caption_mask
    x_last = x * caption_mask

    # l2_loss = torch.cosine_similarity(x_pre, x_last, dim=-1)
    # l2_loss = l2_loss.mean(0).mean(0)
    # l2_loss = 1 - l2_loss
    l2_loss = F.mse_loss(x_last,x_pre)
    return l2_loss

