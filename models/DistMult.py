#coding:utf-8
import numpy as np
import tensorflow as tf
from .Model import Model

class DistMult(Model):
	r'''
	DistMult is based on the bilinear model where each relation is represented by a diagonal rather than a full matrix. 
	DistMult enjoys the same scalable property as TransE and it achieves superior performance over TransE.
	'''
	def _calc(self, h, t, r):
		return h * r * t

	def embedding_def(self):
		config = self.get_config()
		self.ent_embeddings = tf.get_variable(name = "ent_embeddings", shape = [config.entTotal, config.hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform = True))
		self.rel_embeddings = tf.get_variable(name = "rel_embeddings", shape = [config.relTotal, config.hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform = True))
		self.parameter_lists = {"ent_embeddings":self.ent_embeddings, \
								"rel_embeddings":self.rel_embeddings}
	def loss_def(self):
		#Obtaining the initial configuration of the model
		config = self.get_config()
		#To get positive triples and negative triples for training
		#To get labels for the triples, positive triples as 1 and negative triples as -1
		#The shapes of h, t, r, y are (batch_size, 1 + negative_ent + negative_rel)
		h, t, r = self.get_all_instance()
		y = self.get_all_labels()
		#Embedding entities and relations of triples
		e_h = tf.nn.embedding_lookup(self.ent_embeddings, h)
		e_t = tf.nn.embedding_lookup(self.ent_embeddings, t)
		e_r = tf.nn.embedding_lookup(self.rel_embeddings, r)
		#Calculating score functions for all positive triples and negative triples
		res = tf.reduce_sum(self._calc(e_h, e_t, e_r), 1, keep_dims = False)
		loss_func = tf.reduce_mean(tf.nn.softplus(- y * res))
		regul_func = tf.reduce_mean(e_h ** 2) + tf.reduce_mean(e_t ** 2) + tf.reduce_mean(e_r ** 2)
		#Calculating loss to get what the framework will optimize
		self.loss =  loss_func + config.lmbda * regul_func

	def predict_def(self):
		config = self.get_config()
		predict_h, predict_t, predict_r = self.get_predict_instance()
		predict_h_e = tf.nn.embedding_lookup(self.ent_embeddings, predict_h)
		predict_t_e = tf.nn.embedding_lookup(self.ent_embeddings, predict_t)
		predict_r_e = tf.nn.embedding_lookup(self.rel_embeddings, predict_r)
		self.predict = -tf.reduce_sum(self._calc(predict_h_e, predict_t_e, predict_r_e), 1, keep_dims = True)
