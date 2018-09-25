import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .Model import Model

class Analogy(Model):
	def __init__(self,config):
		super(Analogy,self).__init__(config)
		self.ent_re_embeddings=nn.Embedding(self.config.entTotal,self.config.hidden_size/2)
		self.ent_im_embeddings=nn.Embedding(self.config.entTotal,self.config.hidden_size/2)
		self.rel_re_embeddings=nn.Embedding(self.config.relTotal,self.config.hidden_size/2)
		self.rel_im_embeddings=nn.Embedding(self.config.relTotal,self.config.hidden_size/2)
		self.ent_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size)
		self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size)
		self.softplus=nn.Softplus().cuda()
		self.init_weights()
	def init_weights(self):
		nn.init.xavier_uniform(self.ent_re_embeddings.weight.data)
		nn.init.xavier_uniform(self.ent_im_embeddings.weight.data)
		nn.init.xavier_uniform(self.rel_re_embeddings.weight.data)
		nn.init.xavier_uniform(self.rel_im_embeddings.weight.data)
		nn.init.xavier_uniform(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform(self.rel_embeddings.weight.data)

	# score function of Analogy, which is the hybrid of ComplEx and DistMult
	def _calc(self,e_re_h,e_im_h,e_h,e_re_t,e_im_t,e_t,r_re,r_im,r):
		return torch.sum(r_re * e_re_h * e_re_t + r_re * e_im_h * e_im_t + r_im * e_re_h * e_im_t - r_im * e_im_h * e_re_t,1,False) + torch.sum(e_h*e_t*r,1,False)

	def loss_func(self,loss,regul):
		return loss+self.config.lmbda*regul

	def forward(self):
		batch_h,batch_t,batch_r=self.get_all_instance()
		batch_y=self.get_all_labels()
		e_re_h=self.ent_re_embeddings(batch_h)
		e_im_h=self.ent_im_embeddings(batch_h)
		e_h = self.ent_embeddings(batch_h)
		e_re_t=self.ent_re_embeddings(batch_t)
		e_im_t=self.ent_im_embeddings(batch_t)
		e_t=self.ent_embeddings(batch_t)
		r_re=self.rel_re_embeddings(batch_r)
		r_im=self.rel_im_embeddings(batch_r)
		r = self.rel_embeddings(batch_r)
		y=batch_y
		res=self._calc(e_re_h,e_im_h,e_h,e_re_t,e_im_t,e_t,r_re,r_im,r)
		loss = torch.mean(self.softplus(- y * res))
		regul= torch.mean(e_re_h**2)+torch.mean(e_im_h**2)*torch.mean(e_h**2)+torch.mean(e_re_t**2)+torch.mean(e_im_t**2)+torch.mean(e_t**2)+torch.mean(r_re**2)+torch.mean(r_im**2)+torch.mean(r**2)
		#Calculating loss to get what the framework will optimize
		loss =  self.loss_func(loss,regul)
		return loss
	def predict(self, predict_h, predict_t, predict_r):
		p_re_h=self.ent_re_embeddings(Variable(torch.from_numpy(predict_h)).cuda())
		p_re_t=self.ent_re_embeddings(Variable(torch.from_numpy(predict_t)).cuda())
		p_re_r=self.rel_re_embeddings(Variable(torch.from_numpy(predict_r)).cuda())
		p_im_h=self.ent_im_embeddings(Variable(torch.from_numpy(predict_h)).cuda())
		p_im_t=self.ent_im_embeddings(Variable(torch.from_numpy(predict_t)).cuda())
		p_im_r=self.rel_im_embeddings(Variable(torch.from_numpy(predict_r)).cuda())
		p_h=self.ent_im_embeddings(Variable(torch.from_numpy(predict_h)).cuda())
		p_t=self.ent_im_embeddings(Variable(torch.from_numpy(predict_t)).cuda())
		p_r=self.rel_im_embeddings(Variable(torch.from_numpy(predict_r)).cuda())
		p_score = -self._calc(p_re_h, p_im_h, p_h, p_re_t, p_im_t, p_t, p_re_r, p_im_r, p_r)
		return p_score.cpu()
