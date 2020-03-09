#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PyTorchTrainDataset(Dataset):

    def __init__(self, head, tail, rel, ent_total, rel_total, sampling_mode = 'normal', bern_flag = False, filter_flag = True, neg_ent = 1, neg_rel = 0):
        # triples
        self.head = head
        self.tail = tail
        self.rel = rel
        # total numbers of entities, relations, and triples
        self.rel_total = rel_total
        self.ent_total = ent_total
        self.len = len(head)
        # the sampling mode
        self.sampling_mode = sampling_mode
        # the number of negative examples
        self.neg_ent = neg_ent
        self.neg_rel = neg_rel
        self.bern_flag = bern_flag
        self.filter_flag = filter_flag
        if self.sampling_mode == "normal":
        	self.cross_sampling_flag = None
        else:
        	self.cross_sampling_flag = 0
        self.__count_htr()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return (self.head[idx], self.tail[idx], self.rel[idx])

	def collate_fn(self, data):
		batch_data = {}

		if self.sampling_mode == "normal":
			batch_data['mode'] = "normal"
			batch_h_neg = []
			batch_t_neg = []
			batch_r_neg = []
			for item in data:
				neg_head, neg_tail = self.__normal_batch(item[0], item[1], item[2], self.neg_ent)
				neg_rel = self.__rel_batch(item[0], item[1], item[2], self.neg_rel)
				batch_h_neg.append(neg_head)
				batch_t_neg.append(np.ones(len(neg_head), dtype = np.int) * item[1])
				batch_r_neg.append(np.ones(len(neg_head), dtype = np.int) * item[2])
				batch_h_neg.append(np.ones(len(neg_tail), dtype = np.int) * item[0])
				batch_t_neg.append(neg_tail)
				batch_r_neg.append(np.ones(len(neg_tail), dtype = np.int) * item[2])
				batch_h_neg.append(np.ones(len(neg_rel), dtype = np.int) * item[0])
				batch_t_neg.append(np.ones(len(neg_rel), dtype = np.int) * item[1])
				batch_r_neg.append(neg_rel)
			batch_h_neg = np.concatenate(batch_h_neg).reshape(-1, self.neg_ent + self.neg_rel)
			batch_t_neg = np.concatenate(batch_t_neg).reshape(-1, self.neg_ent + self.neg_rel)
			batch_r_neg = np.concatenate(batch_r_neg).reshape(-1, self.neg_ent + self.neg_rel)

			batch_h = np.concatenate([np.array([item[0] for item in data]).resahpe(-1, 1), batch_h_neg], -1).transpose()
			batch_t = np.concatenate([np.array([item[1] for item in data]).resahpe(-1, 1), batch_t_neg], -1).transpose()
			batch_r = np.concatenate([np.array([item[2] for item in data]).resahpe(-1, 1), batch_r_neg], -1).transpose()
			batch_y = np.concatenate([np.ones(len(data), 1), np.zeros(batch_r_neg.shape)], -1).transpose()
		else:
			self.cross_sampling_flag = 1 - self.cross_sampling_flag
			batch_data['mode'] = "cross"
			if self.cross_sampling_flag == 0:
				batch_h = np.array([item[0] for item in data])
				batch_t = np.array([item[1] for item in data])
				batch_r = np.array([item[2] for item in data])
				batch_y = np.ones(len(data), 1)

				batch_h_neg = []
				for item in data:
					neg_head = self.__head_batch(item[0], item[1], item[2], self.neg_ent)
					batch_h_neg.append(neg_head)
				batch_h_neg = np.concatenate(batch_h_neg).reshape(-1, self.neg_ent)

				batch_h = np.concatenate([batch_h.reshape(-1, 1), batch_h_neg], -1).transpose()
				batch_y = np.concatenate([batch_y, np.zeros(batch_h_neg.shape)], -1).transpose()
			else:
				batch_h = np.array([item[0] for item in data]) 
				batch_t = np.array([item[1] for item in data])
				batch_r = np.array([item[2] for item in data])
				batch_y = np.ones(len(data), 1)

				batch_t_neg = []
				for item in data:
					neg_tail = self.__tail_batch(item[0], item[1], item[2], self.neg_ent)
					batch_t_neg.append(neg_tail)
				batch_t_neg = np.concatenate(batch_t_neg).reshape(-1, self.neg_ent)

				batch_t = np.concatenate([batch_t.reshape(-1, 1), batch_t_neg], -1).transpose()
				batch_y = np.concatenate([batch_y, np.zeros(batch_t_neg.shape)], -1).transpose()

		batch_data['batch_h'] = batch_h.reshape(-1)
		batch_data['batch_t'] = batch_t.reshape(-1)
		batch_data['batch_r'] = batch_r.reshape(-1)
		batch_data['batch_y'] = batch_y.reshape(-1)
		return batch_data

    def __count_htr(self):
        
		self.h_of_tr = {}
		self.t_of_hr = {}
		self.r_of_ht = {}
		self.h_of_r = {}
		self.t_of_r = {}
		self.freqRel = {}
		self.lef_mean = {}
		self.rig_mean = {}

		triples = zip(self.head, self.tail, self.rel)
        for h, t, r in triples:
            if (h, r) not in self.t_of_hr:
                self.t_of_hr[(h, r)] = []
            self.t_of_hr[(h, r)].append(t)
            if (t, r) not in self.h_of_tr:
                self.h_of_tr[(t, r)] = []
            self.h_of_tr[(t, r)].append(h)
            if (h, t) not in self.r_of_ht:
            	self.r_of_ht[(h, t)] = []
            self.r_of_ht[(h, t)].append(r)
            if r not in self.freqRel:
            	self.freqRel[r] = 0
            	self.h_of_r[r] = {}
            	self.t_of_r[r] = {}
            self.freqRel[r] += 1.0
            self.h_of_r[r][h] = 1
            self.t_of_r[r][t] = 1

        for t, r in self.h_of_tr:
            self.h_of_tr[(t, r)] = np.array(list(set(self.h_of_tr[(t, r)])))
        for h, r in self.t_of_hr:
            self.t_of_hr[(h, r)] = np.array(list(set(self.h_of_tr[(h, r)])))
        for h, t in self.r_of_ht:
            self.r_of_ht[(h, t)] = np.array(list(set(self.r_of_ht[(h, t)])))
        for r in range(self.rel_total):
        	self.h_of_r[r] = np.array(list(self.h_of_r[r].keys()))
        	self.t_of_r[r] = np.array(list(self.t_of_r[r].keys()))
        	self.lef_mean[r] = self.freqRel[r] / len(self.h_of_r[r])
        	self.rig_mean[r] = self.freqRel[r] / len(self.t_of_r[r])

    def __corrupt_head(self, t, r, num_max = 1):
        tmp = np.random.randint(self.rel_total, size = num_max)
        if not self.filter_flag:
        	return tmp
        mask = np.in1d(tmp, self.h_of_tr[(h, t)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def __corrupt_tail(self, h, r, num_max = 1):
        tmp = np.random.randint(self.rel_total, size = num_max)
        if not self.filter_flag:
        	return tmp
        mask = np.in1d(tmp, self.t_of_hr[(h, t)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def __corrupt_rel(self, h, t, num_max = 1):
        tmp = np.random.randint(self.rel_total, size = num_max)
        if not self.filter_flag:
        	return tmp
        mask = np.in1d(tmp, self.r_of_ht[(h, t)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def __normal_batch(self, h, t, r, neg_size):
        neg_list_h = []
        neg_list_t = []

        if self.bern_flag:
        	neg_size_h = neg_size * right_mean[trainList[i].r] / (right_mean[trainList[i].r] + left_mean[trainList[i].r])
        	neg_size_t = neg_size - neg_size_h
        else:
        	neg_size_h = neg_size
        	neg_size_t = neg_size

        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp_h = self.__corrupt_head(t, r, num_max = neg_size_h * 2):
            neg_tmp_t = self.__corrupt_tail(h, r, num_max = neg_size_t * 2):
            neg_list_h.append(neg_tmp_h)
            neg_list_t.append(neg_tmp_t)
            neg_cur_size += (len(neg_tmp_h) + len(neg_tmp_t))

        neg_list_h = np.concatenate(neg_list_h)
        neg_list_t = np.concatenate(neg_list_t)

        if (len(neg_list_h) <= neg_size_h):
            return neg_list_h, neg_list_t[:neg_size - len(neg_list_h)]
        if (len(neg_list_t) <= neg_size_t):
            return neg_list_h[:neg_size - len(neg_list_t)], neg_list_t
        else:
            return neg_list_h[:neg_size_h], neg_list_t[:neg_size_t]

    def __head_batch(self, h, t, r, neg_size):
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.__corrupt_head(t, r, num_max = neg_size * 2):
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]

    def __tail_batch(self, h, t, r, neg_size):
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.__corrupt_tail(h, r, num_max = neg_size * 2):
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]

    def __rel_batch(self, h, t, r, neg_size):
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.__corrupt_rel(h, t, num_max = neg_size * 2):
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]


class PyTorchTrainDataLoader(DataLoader):

	def __init__(self, 
		in_path = "./", 
		batch_size = None, 
		nbatches = None, 
		threads = 8, 
		sampling_mode = "normal", 
		bern_flag = False, 
		filter_flag = True, 
		neg_ent = 1, 
		neg_rel = 0, 
		shuffle = True, 
		drop_last = False):

		dataset = self.__construct_dataset(in_path, sampling_mode, bern_flag, filter_flag, neg_ent, neg_rel)

		if self.batch_size == None:
			self.batch_size = self.tripleTotal // self.nbatches
		if self.nbatches == None:
			self.nbatches = self.tripleTotal // self.batch_size
		self.batch_seq_size = self.batch_size * (1 + self.negative_ent + self.negative_rel)

		super(PyTorchTrainDataLoader, self).__init__(
			dataset = dataset,
        	batch_size = self.batch_size,
        	shuffle = shuffle,
        	pin_memory = True,
        	num_workers = threads,
        	collate_fn = dataset.collate_fn,
        	drop_last = drop_last)

	def __construct_dataset(self, in_path, sampling_mode, bern_flag, filter_flag, neg_ent, neg_rel):
		f = open(in_path + "entity2id.txt", "r")
		ent_total = (int)(f.readline())
		f.close()

		f = open(in_path + "relation2id.txt", "r")
		rel_total = (int)(f.readline())
		f.close()

		head = []
		tail = []
		rel = []

		f = open(in_path + "train2id.txt", "r")
		triples_total = (int)(f.readline())
		for index in range(triples_total):
			h,t,r = f.readline().strip().splt()
			head.append((int)(h))
			tail.append((int)(t))
			rel.append((int)(r))
		f.close()

		dataset = PyTorchTrainDataset(np.array(head), np.array(tail), np.array(rel), ent_total, rel_total, sampling_mode, bern_flag, filter_flag, neg_ent, neg_rel)
		return dataset

