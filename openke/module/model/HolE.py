#coding:utf-8
import numpy as np
import tensorflow as tf
from .Model import Model

class HolE(Model):

	def __init__(self, ent_tot, rel_tot, dim = 100):
		super(HolE, self).__init__(ent_tot, rel_tot)

		self.dim = dim
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)

		nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

	def _cconv(self, a, b):
		return tf.ifft(tf.fft(a) * tf.fft(b)).real

	def _ccorr(self, a, b):
		a = tf.cast(a, tf.complex64)
		b = tf.cast(b, tf.complex64)
		return tf.real(tf.ifft(tf.conj(tf.fft(a)) * tf.fft(b)))


	def _calc(self, h, t, r):
		return torch.sum(h * t * r, -1)

	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		score = self._calc(h ,t, r)
		return score

	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3
		return regul

	def predict(self, data):
		score = -self.forward(data)
		return score.cpu().data.numpy()








	r'''
	HolE employs circular correlations to create compositional representations. 
	HolE can capture rich interactions but simultaneously remains efficient to compute.
	'''
	def _calc(self, head, tail, rel):
		relation_mention = tf.nn.l2_normalize(rel, 1)
		entity_mention = self._ccorr(head, tail)
		return -tf.sigmoid(tf.reduce_sum(relation_mention * entity_mention, 1, keep_dims = True))



	def loss_def(self):
		#Obtaining the initial configuration of the model
		config = self.get_config()
		#To get positive triples and negative triples for training
		#The shapes of pos_h, pos_t, pos_r are (batch_size, 1)
		#The shapes of neg_h, neg_t, neg_r are (batch_size, negative_ent + negative_rel)
		pos_h, pos_t, pos_r = self.get_positive_instance(in_batch = True)
		neg_h, neg_t, neg_r = self.get_negative_instance(in_batch = True)
		#Embedding entities and relations of triples, e.g. pos_h_e, pos_t_e and pos_r_e are embeddings for positive triples
		pos_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, pos_h), [-1, config.hidden_size])
		pos_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, pos_t), [-1, config.hidden_size])
		pos_r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings, pos_r), [-1, config.hidden_size])
		neg_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, neg_h), [-1, config.hidden_size])
		neg_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, neg_t), [-1, config.hidden_size])
		neg_r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings, neg_r), [-1, config.hidden_size])
		#Calculating score functions for all positive triples and negative triples 
		#The shape of _p_score is (batch_size, 1, 1)
		#The shape of _n_score is (batch_size, negative_ent + negative_rel, 1)
		_p_score = tf.reshape(self._calc(pos_h_e, pos_t_e, pos_r_e), [-1, 1])
		_n_score = tf.reshape(self._calc(neg_h_e, neg_t_e, neg_r_e), [-1, config.negative_rel + config.negative_ent])
		#The shape of p_score is (batch_size, 1)
		#The shape of n_score is (batch_size, 1)
		p_score =  _p_score
		n_score =  tf.reduce_mean(_n_score, 1, keep_dims = True)
		#Calculating loss to get what the framework will optimize
		self.loss = tf.reduce_sum(tf.maximum(p_score - n_score + config.margin, 0))

	def predict_def(self):
		config = self.get_config()
		predict_h, predict_t, predict_r = self.get_predict_instance()
		predict_h_e = tf.nn.embedding_lookup(self.ent_embeddings, predict_h)
		predict_t_e = tf.nn.embedding_lookup(self.ent_embeddings, predict_t)
		predict_r_e = tf.nn.embedding_lookup(self.rel_embeddings, predict_r)
		self.predict = tf.reduce_sum(self._calc(predict_h_e, predict_t_e, predict_r_e), 1, keep_dims = True)
