import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model
from .TransE import TransE

class TransR(Model):
	def __init__(self, config):
		super(TransR, self).__init__(config)
		self.ent_embeddings = nn.Embedding(self.config.entTotal, self.config.ent_size)
		self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.rel_size)
		self.transfer_matrix = nn.Embedding(self.config.relTotal, self.config.ent_size * self.config.rel_size)
		self.criterion = nn.MarginRankingLoss(self.config.margin, False)
		self.init_weights()
		
	def init_weights(self):
		if self.config.pretrain_model == None:
			raise Exception("[ERROR] Pretrain model doesn't exist!")
		self.ent_embeddings.weight.data = self.config.pretrain_model['ent_embeddings.weight']
		self.rel_embeddings.weight.data = self.config.pretrain_model['rel_embeddings.weight']
		identity = torch.zeros(self.config.rel_size, self.config.ent_size)
		for i in range(self.config.rel_size):
			identity[i][i] = 1
			if i == self.config.ent_size - 1:
				break
		identity = identity.view(self.config.ent_size * self.config.rel_size)
		for i in range(self.config.relTotal):
			self.transfer_matrix.weight.data[i] = identity	
	
	def _calc(self, h, t, r):
		return torch.norm(h + r - t, self.config.p_norm, -1)
	
	def _transfer(self, e, r_transfer):
		e = e.view(-1, self.config.ent_size, 1)
		r_transfer = r_transfer.view(-1, self.config.rel_size, self.config.ent_size)
		e = torch.matmul(r_transfer, e)
		e = e.view(-1, self.config.rel_size)
		return e

	def loss(self, p_score, n_score):
		y = Variable(torch.Tensor([-1]).cuda())
		return self.criterion(p_score, n_score, y)
	
	def forward(self):
		h = self.ent_embeddings(self.batch_h)
		t = self.ent_embeddings(self.batch_t)
		r = self.rel_embeddings(self.batch_r)
		r_transfer = self.transfer_matrix(self.batch_r)
		h = self._transfer(h, r_transfer)
		t = self._transfer(t, r_transfer)	
		score = self._calc(h ,t, r)
		p_score = self.get_positive_score(score)
		n_score = self.get_negative_score(score)
		return self.loss(p_score, n_score)	
	
	def predict(self):
		h = self.ent_embeddings(self.batch_h)
		t = self.ent_embeddings(self.batch_t)
		r = self.rel_embeddings(self.batch_r)
		r_transfer = self.transfer_matrix(self.batch_r)
		h = self._transfer(h, r_transfer)
		t = self._transfer(t, r_transfer)
		score = self._calc(h, t, r)
		return score.cpu().data.numpy()	
