import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AdamW
from allennlp.nn.util import batched_index_select
from allennlp.modules import FeedForward
from tqdm import tqdm
import os
import torch.nn.functional as F
from utils import get_devices
from d2l import torch as d2l
from collections import defaultdict
import numpy as np

class RawGlobalPointer(nn.Module):
    def __init__(self, hiddensize, ent_type_size, inner_dim, RoPE=True, tril_mask=True, do_rdrop=False, dropout=0):
        '''
        :param encoder: BERT
        :param ent_type_size: 实体数目
        :param inner_dim: 64
        '''
        super().__init__()
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = hiddensize
        if do_rdrop:
            self.dense = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2))
        else:
            self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)

        self.RoPE = RoPE
        self.trail_mask = tril_mask

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, last_hidden_state,  attention_mask):
        self.device = attention_mask.device
        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]
        outputs = self.dense(last_hidden_state)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]
        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12
        # 排除下三角
        if self.trail_mask:
            mask = torch.tril(torch.ones_like(logits), -1)
            logits = logits - mask * 1e12

        return logits / self.inner_dim ** 0.5


class CLaREModel(nn.Module):
    def __init__(self, encoder_class, args, schema):
        super(CLaREModel, self).__init__()
        self.args = args
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        GlobalPointer = RawGlobalPointer
        ent_type_size = len(args.len_to_label.keys())
        self.mention_detect = GlobalPointer(hiddensize=hiddensize, ent_type_size=ent_type_size, inner_dim=args.inner_dim, RoPE=True, tril_mask=False).to(args.device)
        self.s_o_rel = GlobalPointer(hiddensize=hiddensize, ent_type_size=len(schema), inner_dim=args.inner_dim, RoPE=True, tril_mask=False).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids=None):
        args = self.args
        if args.model_type == 'roberta':
            outputs = self.encoder(batch_token_ids, batch_mask_ids)[0]
        elif args.model_type == 'albert' or args.model_type == 'bert':
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        so_rel_outputs = self.s_o_rel(outputs, batch_mask_ids)
        return mention_outputs, so_rel_outputs
    
