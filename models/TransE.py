#coding:utf-8
import numpy as np
import tensorflow as tf
from Model import *

class TransE(Model):

	def _calc(self, h, t, r):
		return abs(h + r - t)

	def embedding_def(self):
		#Obtaining the initial configuration of the model
		config = self.get_config()
		#Defining required parameters of the model, including embeddings of entities and relations
		self.ent_embeddings = tf.get_variable(name = "ent_embeddings", shape = [config.entTotal, config.hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		self.rel_embeddings = tf.get_variable(name = "rel_embeddings", shape = [config.relTotal, config.hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))

	def loss_def(self):
		#Obtaining the initial configuration of the model
		config = self.get_config()
		#To get positive triples and negative triples for training
		#The shapes of pos_h, pos_t, pos_r are (batch_size, 1)
		#The shapes of neg_h, neg_t, neg_r are (batch_size, negative_ent + negative_rel)
		pos_h, pos_t, pos_r = self.get_positive_instance(in_batch = True)
		neg_h, neg_t, neg_r = self.get_negative_instance(in_batch = True)
		#Embedding entities and relations of triples, e.g. p_h, p_t and p_r are embeddings for positive triples
		p_h = tf.nn.embedding_lookup(self.ent_embeddings, pos_h)
		p_t = tf.nn.embedding_lookup(self.ent_embeddings, pos_t)
		p_r = tf.nn.embedding_lookup(self.rel_embeddings, pos_r)
		n_h = tf.nn.embedding_lookup(self.ent_embeddings, neg_h)
		n_t = tf.nn.embedding_lookup(self.ent_embeddings, neg_t)
		n_r = tf.nn.embedding_lookup(self.rel_embeddings, neg_r)
		#Calculating score functions for all positive triples and negative triples
		#The shape of _p_score is (batch_size, 1, hidden_size)
		#The shape of _n_score is (batch_size, negative_ent + negative_rel, hidden_size)
		_p_score = self._calc(p_h, p_t, p_r)
		_n_score = self._calc(n_h, n_t, n_r)
		#The shape of p_score is (batch_size, 1)
		#The shape of n_score is (batch_size, 1)
		p_score =  tf.reduce_sum(tf.reduce_mean(_p_score, 1, keep_dims = False), 1, keep_dims = True)
		n_score =  tf.reduce_sum(tf.reduce_mean(_n_score, 1, keep_dims = False), 1, keep_dims = True)
		#Calculating loss to get what the framework will optimize
		self.loss = tf.reduce_sum(tf.maximum(p_score - n_score + config.margin, 0))
