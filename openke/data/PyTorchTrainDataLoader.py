#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
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
		self.tri_total = len(head)
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
		return self.tri_total

	def __getitem__(self, idx):
		return (self.head[idx], self.tail[idx], self.rel[idx])

	def collate_fn(self, data):
		batch_data = {}
		if self.sampling_mode == "normal":
			batch_data['mode'] = "normal"
			batch_h = np.array([item[0] for item in data]).reshape(-1, 1)
			batch_t = np.array([item[1] for item in data]).reshape(-1, 1)
			batch_r = np.array([item[2] for item in data]).reshape(-1, 1)
			batch_h = np.repeat(batch_h, 1 + self.neg_ent + self.neg_rel, axis = -1)
			batch_t = np.repeat(batch_t, 1 + self.neg_ent + self.neg_rel, axis = -1)
			batch_r = np.repeat(batch_r, 1 + self.neg_ent + self.neg_rel, axis = -1)
			for index, item in enumerate(data):
				last = 1
				if self.neg_ent > 0:
					neg_head, neg_tail = self.__normal_batch(item[0], item[1], item[2], self.neg_ent)
					if len(neg_head) > 0:
						batch_h[index][last:last + len(neg_head)] = neg_head
						last += len(neg_head)
					if len(neg_tail) > 0:
						batch_t[index][last:last + len(neg_tail)] = neg_tail
						last += len(neg_tail)
				if self.neg_rel > 0:
					neg_rel = self.__rel_batch(item[0], item[1], item[2], self.neg_rel)
					batch_r[index][last:last + len(neg_rel)] = neg_rel
			batch_h = batch_h.transpose()
			batch_t = batch_t.transpose()
			batch_r = batch_r.transpose()
		else:
			self.cross_sampling_flag = 1 - self.cross_sampling_flag
			if self.cross_sampling_flag == 0:
				batch_data['mode'] = "head_batch"
				batch_h = np.array([[item[0]] for item in data])
				batch_t = np.array([item[1] for item in data])
				batch_r = np.array([item[2] for item in data])
				batch_h = np.repeat(batch_h, 1 + self.neg_ent, axis = -1)
				for index, item in enumerate(data):
					neg_head = self.__head_batch(item[0], item[1], item[2], self.neg_ent)
					batch_h[index][1:] = neg_head
				batch_h = batch_h.transpose()
			else:
				batch_data['mode'] = "tail_batch"
				batch_h = np.array([item[0] for item in data]) 
				batch_t = np.array([[item[1]] for item in data])
				batch_r = np.array([item[2] for item in data])
				batch_t = np.repeat(batch_t, 1 + self.neg_ent, axis = -1)
				for index, item in enumerate(data):
					neg_tail = self.__tail_batch(item[0], item[1], item[2], self.neg_ent)
					batch_t[index][1:] = neg_tail
				batch_t = batch_t.transpose()

		batch_y = np.concatenate([np.ones((len(data), 1)), np.zeros((len(data), self.neg_ent + self.neg_rel))], -1).transpose()
		batch_data['batch_h'] = batch_h.squeeze()
		batch_data['batch_t'] = batch_t.squeeze()
		batch_data['batch_r'] = batch_r.squeeze()
		batch_data['batch_y'] = batch_y.squeeze()
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
			self.t_of_hr[(h, r)] = np.array(list(set(self.t_of_hr[(h, r)])))
		for h, t in self.r_of_ht:
			self.r_of_ht[(h, t)] = np.array(list(set(self.r_of_ht[(h, t)])))
		for r in range(self.rel_total):
			self.h_of_r[r] = np.array(list(self.h_of_r[r].keys()))
			self.t_of_r[r] = np.array(list(self.t_of_r[r].keys()))
			self.lef_mean[r] = self.freqRel[r] / len(self.h_of_r[r])
			self.rig_mean[r] = self.freqRel[r] / len(self.t_of_r[r])

	def __corrupt_head(self, t, r, num_max = 1):
		tmp = torch.randint(low = 0, high = self.ent_total, size = (num_max, )).numpy()
		if not self.filter_flag:
			return tmp
		mask = np.in1d(tmp, self.h_of_tr[(t, r)], assume_unique=True, invert=True)
		neg = tmp[mask]
		return neg

	def __corrupt_tail(self, h, r, num_max = 1):
		tmp = torch.randint(low = 0, high = self.ent_total, size = (num_max, )).numpy()
		if not self.filter_flag:
			return tmp
		mask = np.in1d(tmp, self.t_of_hr[(h, r)], assume_unique=True, invert=True)
		neg = tmp[mask]
		return neg

	def __corrupt_rel(self, h, t, num_max = 1):
		tmp = torch.randint(low = 0, high = self.rel_total, size = (num_max, )).numpy()
		if not self.filter_flag:
			return tmp
		mask = np.in1d(tmp, self.r_of_ht[(h, t)], assume_unique=True, invert=True)
		neg = tmp[mask]
		return neg

	def __normal_batch(self, h, t, r, neg_size):
		neg_size_h = 0
		neg_size_t = 0
		prob = self.rig_mean[r] / (self.rig_mean[r] + self.lef_mean[r]) if self.bern_flag else 0.5
		for i in range(neg_size):
			if random.random() < prob:
				neg_size_h += 1
			else:
				neg_size_t += 1

		neg_list_h = []
		neg_cur_size = 0
		while neg_cur_size < neg_size_h:
			neg_tmp_h = self.__corrupt_head(t, r, num_max = (neg_size_h - neg_cur_size) * 2)
			neg_list_h.append(neg_tmp_h)
			neg_cur_size += len(neg_tmp_h)
		if neg_list_h != []:
			neg_list_h = np.concatenate(neg_list_h)
			
		neg_list_t = []
		neg_cur_size = 0
		while neg_cur_size < neg_size_t:
			neg_tmp_t = self.__corrupt_tail(h, r, num_max = (neg_size_t - neg_cur_size) * 2)
			neg_list_t.append(neg_tmp_t)
			neg_cur_size += len(neg_tmp_t)
		if neg_list_t != []:
			neg_list_t = np.concatenate(neg_list_t)

		return neg_list_h[:neg_size_h], neg_list_t[:neg_size_t]

	def __head_batch(self, h, t, r, neg_size):
		# return torch.randint(low = 0, high = self.ent_total, size = (neg_size, )).numpy()
		neg_list = []
		neg_cur_size = 0
		while neg_cur_size < neg_size:
			neg_tmp = self.__corrupt_head(t, r, num_max = (neg_size - neg_cur_size) * 2)
			neg_list.append(neg_tmp)
			neg_cur_size += len(neg_tmp)
		return np.concatenate(neg_list)[:neg_size]

	def __tail_batch(self, h, t, r, neg_size):
		# return torch.randint(low = 0, high = self.ent_total, size = (neg_size, )).numpy()
		neg_list = []
		neg_cur_size = 0
		while neg_cur_size < neg_size:
			neg_tmp = self.__corrupt_tail(h, r, num_max = (neg_size - neg_cur_size) * 2)
			neg_list.append(neg_tmp)
			neg_cur_size += len(neg_tmp)
		return np.concatenate(neg_list)[:neg_size]

	def __rel_batch(self, h, t, r, neg_size):
		neg_list = []
		neg_cur_size = 0
		while neg_cur_size < neg_size:
			neg_tmp = self.__corrupt_rel(h, t, num_max = (neg_size - neg_cur_size) * 2)
			neg_list.append(neg_tmp)
			neg_cur_size += len(neg_tmp)
		return np.concatenate(neg_list)[:neg_size]
	
	def set_sampling_mode(self, sampling_mode):
		self.sampling_mode = sampling_mode

	def set_ent_neg_rate(self, rate):
		self.neg_ent = rate

	def set_rel_neg_rate(self, rate):
		self.neg_rel = rate

	def set_bern_flag(self, bern_flag):
		self.bern_flag = bern_flag

	def set_filter_flag(self, filter_flag):
		self.filter_flag = filter_flag

	def get_ent_tot(self):
		return self.ent_total

	def get_rel_tot(self):
		return self.rel_total

	def get_tri_tot(self):
		return self.tri_total

class PyTorchTrainDataLoader(DataLoader):

	def __init__(self, 
		in_path = None, 
		tri_file = None,
		ent_file = None,
		rel_file = None,
		batch_size = None, 
		nbatches = None, 
		threads = 8, 
		sampling_mode = "normal", 
		bern_flag = False, 
		filter_flag = True, 
		neg_ent = 1, 
		neg_rel = 0, 
		shuffle = True, 
		drop_last = True):

		self.in_path = in_path
		self.tri_file = tri_file
		self.ent_file = ent_file
		self.rel_file = rel_file
		if in_path != None:
			self.tri_file = in_path + "train2id.txt"
			self.ent_file = in_path + "entity2id.txt"
			self.rel_file = in_path + "relation2id.txt"

		dataset = self.__construct_dataset(sampling_mode, bern_flag, filter_flag, neg_ent, neg_rel)

		self.batch_size = batch_size
		self.nbatches = nbatches
		if batch_size == None:
			self.batch_size = dataset.get_tri_tot() // nbatches
		if nbatches == None:
			self.nbatches = dataset.get_tri_tot() // batch_size

		super(PyTorchTrainDataLoader, self).__init__(
			dataset = dataset,
			batch_size = self.batch_size,
			shuffle = shuffle,
			pin_memory = True,
			num_workers = threads,
			collate_fn = dataset.collate_fn,
			drop_last = drop_last)

	def __construct_dataset(self, sampling_mode, bern_flag, filter_flag, neg_ent, neg_rel):
		f = open(self.ent_file, "r")
		ent_total = (int)(f.readline())
		f.close()

		f = open(self.rel_file, "r")
		rel_total = (int)(f.readline())
		f.close()

		head = []
		tail = []
		rel = []

		f = open(self.tri_file, "r")
		triples_total = (int)(f.readline())
		for index in range(triples_total):
			h,t,r = f.readline().strip().split()
			head.append((int)(h))
			tail.append((int)(t))
			rel.append((int)(r))
		f.close()

		dataset = PyTorchTrainDataset(np.array(head), np.array(tail), np.array(rel), ent_total, rel_total, sampling_mode, bern_flag, filter_flag, neg_ent, neg_rel)
		return dataset

	"""interfaces to set essential parameters"""

	def set_sampling_mode(self, sampling_mode):
		self.dataset.set_sampling_mode(sampling_mode)

	def set_work_threads(self, work_threads):
		self.num_workers = work_threads

	def set_nbatches(self, nbatches):
		self.nbatches = nbatches
		self.batch_size = self.tripleTotal // self.nbatches

	def set_batch_size(self, batch_size):
		self.batch_size = batch_size
		self.nbatches = self.tripleTotal // self.batch_size

	def set_ent_neg_rate(self, rate):
		self.dataset.set_ent_neg_rate(rate)

	def set_rel_neg_rate(self, rate):
		self.dataset.set_rel_neg_rate(rate)

	def set_bern_flag(self, bern_flag):
		self.dataset.set_bern_flag(bern_flag)

	def set_filter_flag(self, filter_flag):
		self.dataset.set_filter_flag(filter_flag)
	
	"""interfaces to get essential parameters"""
	
	def get_batch_size(self):
		return self.batch_size

	def get_ent_tot(self):
		return self.dataset.get_ent_tot()

	def get_rel_tot(self):
		return self.dataset.get_rel_tot()

	def get_triple_tot(self):
		return self.dataset.get_tri_tot()