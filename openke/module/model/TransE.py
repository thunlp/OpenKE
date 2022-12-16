import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model
import torch.distributed as dist


class parallel_normalize(torch.autograd.Function):
	def __init__(self):
		super(parallel_normalize, self).__init__()

	@staticmethod
	def forward(ctx, input):
		sum_of_square = (input * input).sum(-1) 
		dist.all_reduce(sum_of_square, dist.ReduceOp.SUM,)
		input_norm = sum_of_square ** 0.5
		inverse_l2_norm = 1 / input_norm
		input_norm_mask = input_norm < 1
		ctx.save_for_backward(inverse_l2_norm, input, input_norm_mask)
		return input * (inverse_l2_norm).unsqueeze(1) 


	@staticmethod
	def backward(ctx, grad_output):
		inverse_l2_norm, input, input_norm_mask = ctx.saved_tensors
		bs, dim = input.shape
		tensor_list = [torch.zeros_like(input)]*2
		dist.all_gather(tensor_list, input)		# communicate half a model is problematic
		input = torch.cat(tensor_list, 1)
		outer = torch.bmm(input.unsqueeze(2), input.unsqueeze(1))
		jacobian = -1 * outer * torch.pow(inverse_l2_norm, 3).unsqueeze(1).unsqueeze(1)
		jacobian_diag = torch.zeros_like(jacobian)
		for i in range(len(jacobian)):
			jacobian_diag[i].fill_diagonal_(inverse_l2_norm[i].item())
		jacobian += jacobian_diag
		jacobian = jacobian[:dim]
		output = torch.bmm(grad_output.unsqueeze(1), jacobian).squeeze()
		return output


class TransE(Model):

	def __init__(self, ent_tot, rel_tot, dim = 100, p_norm = 1, norm_flag = True, margin = None, epsilon = None, world_size=2):
		super(TransE, self).__init__(ent_tot, rel_tot)
		
		self.dim = dim
		self.margin = margin
		self.epsilon = epsilon
		self.norm_flag = norm_flag
		self.p_norm = p_norm

		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim // world_size)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim // world_size)
		self.parallel_normalize = parallel_normalize().apply
		if margin == None or epsilon == None:
			nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
		else:
			self.embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim//world_size]), requires_grad=False
			)
			nn.init.uniform_(
				tensor = self.ent_embeddings.weight.data, 
				a = -self.embedding_range.item(), 
				b = self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_embeddings.weight.data, 
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)

		if margin != None:
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False
			self.margin_flag = True
		else:
			self.margin_flag = False


	def _calc(self, h, t, r, mode):
		if self.norm_flag:
			h = self.parallel_normalize(h)
			r = self.parallel_normalize(r)
			t = self.parallel_normalize(t)
		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
		if mode == 'head_batch':
			score = h + (r - t)
		else:
			score = (h + r) - t
		score = torch.norm(score, self.p_norm, -1).flatten()
		dist.all_reduce(score, dist.ReduceOp.SUM,)
		return score

	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		score = self._calc(h ,t, r, mode)
		if self.margin_flag:
			return self.margin - score
		else:
			return score

	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		regul = (torch.mean(h ** 2) + 
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2)) / 3
		return regul

	def predict(self, data):
		score = self.forward(data)
		if self.margin_flag:
			score = self.margin - score
			return score.cpu().data.numpy()
		else:
			return score.cpu().data.numpy()