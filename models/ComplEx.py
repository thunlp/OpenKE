#coding:utf-8
import numpy as np
import tensorflow as tf
from .Model import Model

class ComplEx(Model):

	def embedding_def(self):
		config = self.get_config()
		self.ent1_embeddings = tf.get_variable(name = "ent1_embeddings", shape = [config.entTotal, config.hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform = True))
		self.rel1_embeddings = tf.get_variable(name = "rel1_embeddings", shape = [config.relTotal, config.hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform = True))
		self.ent2_embeddings = tf.get_variable(name = "ent2_embeddings", shape = [config.entTotal, config.hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform = True))
		self.rel2_embeddings = tf.get_variable(name = "rel2_embeddings", shape = [config.relTotal, config.hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform = True))
		self.parameter_lists = {"ent_re_embeddings":self.ent1_embeddings, \
								"ent_im_embeddings":self.ent2_embeddings, \
								"rel_re_embeddings":self.rel1_embeddings, \
								"rel_im_embeddings":self.rel2_embeddings}
	r'''
	ComplEx extends DistMult by introducing complex-valued embeddings so as to better model asymmetric relations. 
	It is proved that HolE is subsumed by ComplEx as a special case.
	'''
	def _calc(self, e1_h, e2_h, e1_t, e2_t, r1, r2):
		return tf.reduce_sum(e1_h * e1_t * r1 + e2_h * e2_t * r1 + e1_h * e2_t * r2 - e2_h * e1_t * r2, -1, keep_dims = False)

	def loss_def(self):
		#Obtaining the initial configuration of the model
		config = self.get_config()
		#To get positive triples and negative triples for training
		#To get labels for the triples, positive triples as 1 and negative triples as -1
		#The shapes of h, t, r, y are (batch_size, 1 + negative_ent + negative_rel)
		pos_h, pos_t, pos_r = self.get_positive_instance(in_batch = True)
		neg_h, neg_t, neg_r = self.get_negative_instance(in_batch = True)
		pos_y = self.get_positive_labels(in_batch = True)
		neg_y = self.get_negative_labels(in_batch = True)

		p1_h = tf.nn.embedding_lookup(self.ent1_embeddings, pos_h)
		p2_h = tf.nn.embedding_lookup(self.ent2_embeddings, pos_h)
		p1_t = tf.nn.embedding_lookup(self.ent1_embeddings, pos_t)
		p2_t = tf.nn.embedding_lookup(self.ent2_embeddings, pos_t)
		p1_r = tf.nn.embedding_lookup(self.rel1_embeddings, pos_r)
		p2_r = tf.nn.embedding_lookup(self.rel2_embeddings, pos_r)

		n1_h = tf.nn.embedding_lookup(self.ent1_embeddings, neg_h)
		n2_h = tf.nn.embedding_lookup(self.ent2_embeddings, neg_h)
		n1_t = tf.nn.embedding_lookup(self.ent1_embeddings, neg_t)
		n2_t = tf.nn.embedding_lookup(self.ent2_embeddings, neg_t)
		n1_r = tf.nn.embedding_lookup(self.rel1_embeddings, neg_r)
		n2_r = tf.nn.embedding_lookup(self.rel2_embeddings, neg_r)

		_p_score = self._calc(p1_h, p2_h, p1_t, p2_t, p1_r, p2_r)
		_n_score = self._calc(n1_h, n2_h, n1_t, n2_t, n1_r, n2_r)
		print (_n_score.get_shape())
		loss_func = tf.reduce_mean(tf.nn.softplus(- pos_y * _p_score) + tf.nn.softplus(- neg_y * _n_score))
		regul_func = tf.reduce_mean(p1_h ** 2 + p1_t ** 2 + p1_r ** 2 + n1_h ** 2 + n1_t ** 2 + n1_r ** 2 + p2_h ** 2 + p2_t ** 2 + p2_r ** 2 + n2_h ** 2 + n2_t ** 2 + n2_r ** 2) 
		self.loss =  loss_func + config.lmbda * regul_func

	def predict_def(self):
		config = self.get_config()
		predict_h, predict_t, predict_r = self.get_predict_instance()
		predict_h_e1 = tf.nn.embedding_lookup(self.ent1_embeddings, predict_h)
		predict_t_e1 = tf.nn.embedding_lookup(self.ent1_embeddings, predict_t)
		predict_r_e1 = tf.nn.embedding_lookup(self.rel1_embeddings, predict_r)
		predict_h_e2 = tf.nn.embedding_lookup(self.ent2_embeddings, predict_h)
		predict_t_e2 = tf.nn.embedding_lookup(self.ent2_embeddings, predict_t)
		predict_r_e2 = tf.nn.embedding_lookup(self.rel2_embeddings, predict_r)
		self.predict = -self._calc(predict_h_e1, predict_h_e2, predict_t_e1, predict_t_e2, predict_r_e1, predict_r_e2)

