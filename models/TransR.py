import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Model import *
class TransR(Model):
	def __init__(self,config):
		super(TransR,self).__init__(config)
		self.ent_embeddings=nn.Embedding(self.config.entTotal,self.config.ent_size)
		self.rel_embeddings=nn.Embedding(self.config.relTotal,self.config.rel_size)
		self.transfer_matrix=nn.Embedding(self.config.relTotal,self.config.ent_size*self.config.rel_size)
		self.init_weights()
	def init_weights(self):
		nn.init.xavier_uniform(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform(self.rel_embeddings.weight.data)
		nn.init.xavier_uniform(self.transfer_matrix.weight.data)
	def _transfer(self,transfer_matrix,embeddings):
		return torch.matmul(transfer_matrix,embeddings)
	def _calc(self,h,t,r):
		return torch.abs(h+r-t)
	def loss_func(self,p_score,n_score):
		criterion= nn.MarginRankingLoss(self.config.margin,False).cuda()
		y=Variable(torch.Tensor([-1])).cuda()
		loss=criterion(p_score,n_score,y)
		return loss
	def forward(self):
		pos_h,pos_t,pos_r=self.get_postive_instance()
		neg_h,neg_t,neg_r=self.get_negtive_instance()
		p_h_e=self.ent_embeddings(pos_h).view(-1,self.config.ent_size,1)
		p_t_e=self.ent_embeddings(pos_t).view(-1,self.config.ent_size,1)
		p_r_e=self.rel_embeddings(pos_r).view(-1,self.config.rel_size)
		n_h_e=self.ent_embeddings(neg_h).view(-1,self.config.ent_size,1)
		n_t_e=self.ent_embeddings(neg_t).view(-1,self.config.ent_size,1)
		n_r_e=self.rel_embeddings(neg_r).view(-1,self.config.rel_size)
		p_matrix=self.transfer_matrix(pos_r).view(-1,self.config.rel_size,self.config.ent_size)
		n_matrix=self.transfer_matrix(neg_r).view(-1,self.config.rel_size,self.config.ent_size)
		p_h=self._transfer(p_matrix,p_h_e).view(-1,self.config.rel_size)
		p_t=self._transfer(p_matrix,p_t_e).view(-1,self.config.rel_size)
		p_r=p_r_e				
		n_h=self._transfer(n_matrix,n_h_e).view(-1,self.config.rel_size)
		n_t=self._transfer(n_matrix,n_t_e).view(-1,self.config.rel_size)
		n_r=n_r_e
		_p_score = self._calc(p_h, p_t, p_r)
		_n_score = self._calc(n_h, n_t, n_r)
		p_score=torch.sum(_p_score,1)
		n_score=torch.sum(_n_score,1)
		loss=self.loss_func(p_score,n_score)
		return loss
	def predict(self, predict_h, predict_t, predict_r):
		p_h_e=self.ent_embeddings(Variable(torch.from_numpy(predict_h))).view(-1,self.config.ent_size,1)
		p_t_e=self.ent_embeddings(Variable(torch.from_numpy(predict_t))).view(-1,self.config.ent_size,1)
		p_r_e=self.rel_embeddings(Variable(torch.from_numpy(predict_r))).view(-1,self.config.rel_size)
		p_matrix=self.transfer_matrix(Variable(torch.from_numpy(predict_r))).view(-1,self.config.rel_size,self.config.ent_size)
		p_h=self._transfer(p_matrix,p_h_e).view(-1,self.config.rel_size)
		p_t=self._transfer(p_matrix,p_t_e).view(-1,self.config.rel_size)
		p_r=p_r_e				
		_p_score = self._calc(p_h, p_t, p_r)
		p_score=torch.sum(_p_score,1)
		return p_score.cpu()
		
