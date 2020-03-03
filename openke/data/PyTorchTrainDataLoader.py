#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset


class data_set(Dataset):

    def __init__(self, head, tail, rel, ent_total, rel_total, sampling_mode):
        self.head = head
        self.tail = tail
        self.rel = rel
        self.len = len(head)
        self.rel_total = rel_total
        self.ent_total = ent_total
        self.sampling_mode = sampling_mode

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return (self.head[idx], self.tail[idx], self.rel[idx])

    def __corrupt_head(self, t, r, num_max = 1):
        tmp = np.random.randint(self.rel_total, size = num_max)
        mask = np.in1d(tmp, self.h_of_tr[(h, t)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def __corrupt_tail(self, h, r, num_max = 1):
        tmp = np.random.randint(self.rel_total, size = num_max)
        mask = np.in1d(tmp, self.t_of_hr[(h, t)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def __corrupt_rel(self, h, t, num_max = 1):
        tmp = np.random.randint(self.rel_total, size = num_max)
        mask = np.in1d(tmp, self.rel_of_ht[(h, t)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def normal_batch(self, h, t, r, neg_size, bern_flag):
        neg_list_h = []
        neg_list_t = []

        neg_size_h = neg_size * right_mean[trainList[i].r] / (right_mean[trainList[i].r] + left_mean[trainList[i].r])
        neg_size_t = neg_size - neg_size_h

        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp_h = self.__corrupt_head(t, r, num_max = neg_size_h * 2):
            neg_tmp_t = self.__corrupt_tail(h, r, num_max = neg_size_t * 2):
            neg_list_h.append(neg_tmp_h)
            neg_list_t.append(neg_tmp_t)
            neg_cur_size += (len(neg_tmp_h) + len(neg_tmp_t))

        neg_list_h = np.concatenate(neg_list_h)
        neg_list_t = np.concatenate(neg_list_t)
        if (len())
        return

    def head_batch(self, h, t, r, neg_size):
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.__corrupt_head(t, r, num_max = neg_size * 2):
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]

    def tail_batch(self, h, t, r, neg_size):
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.__corrupt_tail(h, r, num_max = neg_size * 2):
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]
