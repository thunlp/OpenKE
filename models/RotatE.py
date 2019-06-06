import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model

import pdb

class RotatE(Model):
	def __init__(self, config):
		super().__init__(config)
		self.embedding_range = nn.Parameter(
			torch.Tensor([(self.config.margin + self.config.epsilon) / self.config.hidden_size]), 
			requires_grad=False
			)
		self.entity_dim = self.config.hidden_size*2 if self.config.ent_double_embedding else self.config.hidden_size
		self.relation_dim = self.config.hidden_size*2 if self.config.rel_double_embedding else self.config.hidden_size

		self.ent_embeddings = nn.Parameter(torch.zeros(self.config.entTotal, self.entity_dim))
		self.rel_embeddings = nn.Parameter(torch.zeros(self.config.relTotal, self.relation_dim))
		# self.ent_embeddings = nn.Embedding(self.config.entTotal, self.entity_dim)
		# self.rel_embeddings = nn.Embedding(self.config.relTotal, self.relation_dim)
		# if not self.config.self_adv:
		# 	self.criterion = nn.MarginRankingLoss(self.config.margin, False)
		# else:
		self.criterion = self.SelfAdv(self.config)
		self.init_weights()
		
	def init_weights(self):
		# nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
		# nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
		nn.init.uniform_(
			tensor=self.ent_embeddings, 
			a=-self.embedding_range.item(), 
			b=self.embedding_range.item()
		)
        
		nn.init.uniform_(
			tensor=self.rel_embeddings, 
			a=-self.embedding_range.item(), 
			b=self.embedding_range.item()
		)

	def _calc(self, h, t, r, head_batch=False):
		pi = 3.14159265358979323846
		# pdb.set_trace()

		re_head, im_head = torch.chunk(h, 2, dim=-1)
		re_tail, im_tail = torch.chunk(t, 2, dim=-1)

		#Make phases of relations uniformly distributed in [-pi, pi]

		phase_relation = r/(self.embedding_range.item()/pi)

		re_relation = torch.cos(phase_relation)
		im_relation = torch.sin(phase_relation)

		re_head = re_head.view(-1, re_relation.shape[0], re_head.shape[-1]).permute(1, 0, 2)
		re_tail = re_tail.view(-1, re_relation.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
		im_head = im_head.view(-1, re_relation.shape[0], im_head.shape[-1]).permute(1, 0, 2)
		im_tail = im_tail.view(-1, re_relation.shape[0], im_tail.shape[-1]).permute(1, 0, 2)
		# re_relation = re_relation.unsqueeze(0).repeat(1, int(re_head.shape[0] / re_relation.shape[0]), 1, 1).squeeze(0)
		# im_relation = im_relation.unsqueeze(0).repeat(1, int(im_head.shape[0] / im_relation.shape[0]), 1, 1).squeeze(0)
		# pdb.set_trace()
		if head_batch:
		    re_score = re_relation * re_tail + im_relation * im_tail
		    im_score = re_relation * im_tail - im_relation * re_tail
		    re_score = re_score - re_head
		    im_score = im_score - im_head
		else:
			re_score = re_head * re_relation - im_head * im_relation
			im_score = re_head * im_relation + im_head * re_relation
			re_score = re_score - re_tail
			im_score = im_score - im_tail

		score = torch.stack([re_score, im_score], dim = 0)
		score = score.norm(dim = 0)

		score = score.sum(dim = 2)
		return score.permute(1, 0).flatten()
	
	def loss(self, p_score, n_score):
		y = Variable(torch.Tensor([-1]).cuda())
		return self.criterion(p_score, n_score, y)

	def forward(self, head_batch=False):
		# pdb.set_trace()
		if self.config.cross_sampling:
			if head_batch:
				h = torch.index_select(
					self.ent_embeddings, 
					dim=0, 
					index=self.batch_h
				)

				t = torch.index_select(
					self.ent_embeddings, 
					dim=0, 
					index=self.batch_t[:self.config.batch_size]
				)
			else:
				h = torch.index_select(
					self.ent_embeddings, 
					dim=0, 
					index=self.batch_h[:self.config.batch_size]
				)

				t = torch.index_select(
					self.ent_embeddings, 
					dim=0, 
					index=self.batch_t
				)
		else:
			h = torch.index_select(
				self.ent_embeddings, 
				dim=0, 
				index=self.batch_h
			)

			t = torch.index_select(
				self.ent_embeddings, 
				dim=0, 
				index=self.batch_t
			)
		r = torch.index_select(
			self.rel_embeddings, 
			dim=0, 
			index=self.batch_r[:self.config.batch_size]
		).unsqueeze(1)
		
		
		# pdb.set_trace()

		score = self._calc(h ,t, r, head_batch)
		# pdb.set_trace()
		p_score = self.get_positive_score(score)
		# if not self.config.self_adv:
		# 	# n_score = self.get_negative_score(score)
		# 	return self.loss(p_score, n_score[])
		# else:
		return self.loss(p_score, score[self.config.batch_size:self.config.batch_seq_size])
		
			
	def predict(self):
		h = torch.index_select(
			self.ent_embeddings, 
			dim=0, 
			index=self.batch_h
		).unsqueeze(1)
		
		r = torch.index_select(
			self.rel_embeddings, 
			dim=0, 
			index=self.batch_r
		).unsqueeze(1)
		
		t = torch.index_select(
			self.ent_embeddings, 
			dim=0, 
			index=self.batch_t
		).unsqueeze(1)
		score = self._calc(h, t, r)
		return score.cpu().data.numpy()	
