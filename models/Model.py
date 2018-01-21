import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Model(nn.Module):

	def __init__(self,config):
		super(Model,self).__init__()
		self.config = config

	def get_postive_instance(self):
		self.postive_h = Variable(torch.from_numpy(self.config.batch_h[0:self.config.batch_size])).cuda()
		self.postive_t = Variable(torch.from_numpy(self.config.batch_t[0:self.config.batch_size])).cuda()
		self.postive_r = Variable(torch.from_numpy(self.config.batch_r[0:self.config.batch_size])).cuda()
		return self.postive_h,self.postive_t,self.postive_r

	def get_negtive_instance(self):
		self.negtive_h = Variable(torch.from_numpy(self.config.batch_h[self.config.batch_size:self.config.batch_seq_size])).cuda()
		self.negtive_t = Variable(torch.from_numpy(self.config.batch_t[self.config.batch_size:self.config.batch_seq_size])).cuda()
		self.negtive_r = Variable(torch.from_numpy(self.config.batch_r[self.config.batch_size:self.config.batch_seq_size])).cuda()
		return self.negtive_h,self.negtive_t,self.negtive_r

	def get_all_instance(self):
		self.batch_h = Variable(torch.from_numpy(self.config.batch_h)).cuda()
		self.batch_t = Variable(torch.from_numpy(self.config.batch_t)).cuda()
		self.batch_r = Variable(torch.from_numpy(self.config.batch_r)).cuda()
		return self.batch_h, self.batch_t, self.batch_r

	def get_all_labels(self):
		self.batch_y=Variable(torch.from_numpy(self.config.batch_y)).cuda()
		return self.batch_y

	def predict(self):
		pass

	def forward(self):
		pass

	def loss_func(self):
		pass

