#coding:utf-8
import numpy as np
import tensorflow as tf
from Model import *

class TransH(Model):

	def _transfer(self, e, n):
		# norm = tf.nn.l2_normalize(n, 1)
		norm = n
		return e - tf.reduce_sum(e * norm, 1, keep_dims = True) * norm

	def _calc(self, h, t, r):
		return abs(h + r - t)

	def embedding_def(self):
		#Obtaining the initial configuration of the model
		config = self.get_config()
		#Defining required parameters of the model, including embeddings of entities and relations, and normal vectors of planes
		self.ent_embeddings = tf.get_variable(name = "ent_embeddings", shape = [config.entTotal, config.hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		self.rel_embeddings = tf.get_variable(name = "rel_embeddings", shape = [config.relTotal, config.hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		self.normal_vector = tf.get_variable(name = "normal_vector", shape = [config.relTotal, config.hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))

	def loss_def(self):
		#Obtaining the initial configuration of the model
		config = self.get_config()
		#To get positive triples and negative triples for training
		#The shapes of pos_h, pos_t, pos_r are (batch_size)
		#The shapes of neg_h, neg_t, neg_r are ((negative_ent + negative_rel) × batch_size)
		pos_h, pos_t, pos_r = self.get_positive_instance(in_batch = False)
		neg_h, neg_t, neg_r = self.get_negative_instance(in_batch = False)
		#Embedding entities and relations of triples, e.g. pos_h_e, pos_t_e and pos_r_e are embeddings for positive triples
		pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, pos_h)
		pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, pos_t)
		pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, pos_r)
		neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, neg_h)
		neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, neg_t)
		neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, neg_r)
		#Getting the required normal vectors of planes to transfer entity embeddings
		pos_norm = tf.nn.embedding_lookup(self.normal_vector, pos_r)
		neg_norm = tf.nn.embedding_lookup(self.normal_vector, neg_r)
		#Calculating score functions for all positive triples and negative triples
		p_h = self._transfer(pos_h_e, pos_norm)
		p_t = self._transfer(pos_t_e, pos_norm)
		p_r = pos_r_e
		n_h = self._transfer(neg_h_e, neg_norm)
		n_t = self._transfer(neg_t_e, neg_norm)
		n_r = neg_r_e
		#Calculating score functions for all positive triples and negative triples
		#The shape of _p_score is (1, batch_size, hidden_size)
		#The shape of _n_score is (negative_ent + negative_rel, batch_size, hidden_size)
		_p_score = self._calc(p_h, p_t, p_r)
		_p_score = tf.reshape(_p_score, [1, -1, config.rel_size])
		_n_score = self._calc(n_h, n_t, n_r)
		_n_score = tf.reshape(_n_score, [config.negative_ent + config.negative_rel, -1, config.rel_size])
		#The shape of p_score is (batch_size, 1)
		#The shape of n_score is (batch_size, 1)
		p_score =  tf.reduce_sum(tf.reduce_mean(_p_score, 0, keep_dims = False), 1, keep_dims = True)
		n_score =  tf.reduce_sum(tf.reduce_mean(_n_score, 0, keep_dims = False), 1, keep_dims = True)
		#Calculating loss to get what the framework will optimize
		self.loss = tf.reduce_sum(tf.maximum(p_score - n_score + config.margin, 0))
