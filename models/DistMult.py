import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model

class DistMult(Model):
	def __init__(self, config):
		super(DistMult, self).__init__(config)
		self.ent_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size)
		self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size)
		self.criterion = nn.Softplus()
		self.init_weights()
		
	def init_weights(self):
		nn.init.xavier_uniform(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform(self.rel_embeddings.weight.data)

	def _calc(self, h, t, r):
		return - torch.sum(h * t * r, -1)
	
	def loss(self, score, regul):
		return torch.mean(self.criterion(score * self.batch_y)) + self.config.lmbda * regul 

	def forward(self):
		h = self.ent_embeddings(self.batch_h)
		t = self.ent_embeddings(self.batch_t)
		r = self.rel_embeddings(self.batch_r)
		score = self._calc(h ,t, r)
		regul = torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)
		return self.loss(score, regul)	

	def predict(self):
		h = self.ent_embeddings(self.batch_h)
		t = self.ent_embeddings(self.batch_t)
		r = self.rel_embeddings(self.batch_r)
		score = self._calc(h, t, r)
		return score.cpu().data.numpy()	
