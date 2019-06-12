import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model

class TransE(Model):
	def __init__(self, config):
		super(TransE, self).__init__(config)
		self.ent_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size)
		self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size)
		self.criterion = nn.MarginRankingLoss(self.config.margin, False)
		self.init_weights()
		
	def init_weights(self):
		nn.init.xavier_uniform(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform(self.rel_embeddings.weight.data)

	def _calc(self, h, t, r):
		return torch.norm(h + r - t, self.config.p_norm, -1)
	
	def loss(self, p_score, n_score):
		y = Variable(torch.Tensor([-1]).cuda())
		return self.criterion(p_score, n_score, y)

	def forward(self):
		h = self.ent_embeddings(self.batch_h)
		t = self.ent_embeddings(self.batch_t)
		r = self.rel_embeddings(self.batch_r)
		score = self._calc(h ,t, r)
		p_score = self.get_positive_score(score)
		n_score = self.get_negative_score(score)
		return self.loss(p_score, n_score)	

	def predict(self):
		h = self.ent_embeddings(self.batch_h)
		t = self.ent_embeddings(self.batch_t)
		r = self.rel_embeddings(self.batch_r)
		score = self._calc(h, t, r)
		return score.cpu().data.numpy()	
