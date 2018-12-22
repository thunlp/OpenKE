import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model

class RESCAL(Model):
	def __init__(self, config):
		super(RESCAL, self).__init__(config)
		self.ent_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size)
		self.rel_matrices = nn.Embedding(self.config.relTotal, self.config.hidden_size * self.config.hidden_size)
		self.criterion = nn.MarginRankingLoss(self.config.margin, False)
		self.init_weights()
		
	def init_weights(self):
		nn.init.xavier_uniform(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform(self.rel_matrices.weight.data)

	def _calc(self, h, t, r):
		t = t.view(-1, self.config.hidden_size, 1)
		r = r.view(-1, self.config.hidden_size, self.config.hidden_size)
		tr = torch.matmul(r, t)
		tr = tr.view(-1, self.config.hidden_size)
		return - torch.sum(h * tr, -1)

	def loss(self, p_score, n_score):
		y = Variable(torch.Tensor([-1]).cuda())
		return self.criterion(p_score, n_score, y)

	def forward(self):
		h = self.ent_embeddings(self.batch_h)
		t = self.ent_embeddings(self.batch_t)
		r = self.rel_matrices(self.batch_r)
		score = self._calc(h ,t, r)
		p_score = self.get_positive_score(score)
		n_score = self.get_negative_score(score)
		return self.loss(p_score, n_score)
	def predict(self):
		h = self.ent_embeddings(self.batch_h)
		t = self.ent_embeddings(self.batch_t)
		r = self.rel_matrices(self.batch_r)
		score = self._calc(h, t, r)
		return score.cpu().data.numpy()	
