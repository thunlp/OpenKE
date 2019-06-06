import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import pdb

class Model(nn.Module):
	def __init__(self, config):
		super(Model, self).__init__()
		self.config = config
		self.batch_h = None
		self.batch_t = None
		self.batch_r = None
		self.batch_y = None
	'''
	def get_positive_instance(self):
		self.positive_h = self.batch_h[0:self.config.batch_size]
		self.positive_t = self.batch_t[0:self.config.batch_size]
		self.positive_r = self.batch_r[0:self.config.batch_size]
		return self.positive_h, self.positive_t, self.positive_r

	def get_negative_instance(self):
		self.negative_h = self.batch_h[self.config.batch_size, self.config.batch_seq_size]
		self.negative_t = self.batch_t[self.config.batch_size, self.config.batch_seq_size]
		self.negative_r = self.batch_r[self.config.batch_size, self.config.batch_seq_size]
		return self.negative_h, self.negative_t, self.negative_r
 	'''
	def get_positive_score(self, score):
		return score[0:self.config.batch_size]

	def get_negative_score(self, score):
		negative_score = score[self.config.batch_size:self.config.batch_seq_size]
		negative_score = negative_score.view(-1, self.config.batch_size)
		negative_score = torch.mean(negative_score, 0)
		return negative_score

	def forward(self):
		raise NotImplementedError
	
	def predict(self):
		raise NotImplementedError	


	class SelfAdv(nn.Module):
		def __init__(self, config):
			super().__init__()
			self.config = config
		
		def forward(self, p_score, n_score, y):
			p_score = self.config.margin - p_score
			n_score = n_score.view(-1, p_score.shape[0]) - self.config.margin
			n_score = self.get_adv_neg_score(n_score)
			return -(F.logsigmoid(p_score).mean() + n_score.mean()) / 2

		def get_adv_neg_score(self, n_score):
			if self.config.self_adv:
				return (F.softmax(-n_score * self.config.adv_temperature, dim=0).detach() 
								* F.logsigmoid(n_score)).sum(dim=0)
			else:
				# pdb.set_trace()
				return F.logsigmoid(n_score).mean(dim=0)