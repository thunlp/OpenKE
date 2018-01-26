import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Model import *
class DistMult(Model):
	def __init__(self,config):
		super(DistMult,self).__init__(config)
		self.ent_embeddings=nn.Embedding(self.config.entTotal,self.config.hidden_size)
		self.rel_embeddings=nn.Embedding(self.config.relTotal,self.config.hidden_size)
		self.softplus=nn.Softplus().cuda()
		self.init_weights()
	def init_weights(self):
		nn.init.xavier_uniform(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform(self.rel_embeddings.weight.data)
	def _calc(self,h,t,r):
		return torch.sum(h*t*r,1,False)
	def loss_func(self,loss,regul):
		return loss+self.config.lmbda*regul
	def forward(self):
		batch_h,batch_t,batch_r=self.get_all_instance()
		batch_y=self.get_all_labels()
		e_h=self.ent_embeddings(batch_h)
		e_t=self.ent_embeddings(batch_t)
		e_r=self.rel_embeddings(batch_r)
		y=batch_y
		res=self._calc(e_h,e_t,e_r)
		tmp=self.softplus(- y * res)
		loss = torch.mean(tmp)
		regul = torch.mean(e_h ** 2) + torch.mean(e_t ** 2) + torch.mean(e_r ** 2)
		#Calculating loss to get what the framework will optimize
		loss =  self.loss_func(loss,regul)
		return loss
	def predict(self, predict_h, predict_t, predict_r):
		p_e_h=self.ent_embeddings(Variable(torch.from_numpy(predict_h)).cuda())
		p_e_t=self.ent_embeddings(Variable(torch.from_numpy(predict_t)).cuda())
		p_e_r=self.rel_embeddings(Variable(torch.from_numpy(predict_r)).cuda())
		p_score=-self._calc(p_e_h,p_e_t,p_e_r)
		return p_score.cpu()
		
