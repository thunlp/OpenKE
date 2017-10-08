#coding:utf-8
import numpy as np
import tensorflow as tf
from Model import *

class TransD(Model):

	def _transfer(self, e, t, r):
		return e + tf.reduce_sum(e * t, 1, keep_dims = True) * r

	def _calc(self, h, t, r):
		return abs(h + r - t)

	def embedding_def(self):
		#Obtaining the initial configuration of the model
		config = self.get_config()
		#Defining required parameters of the model, including embeddings of entities and relations, entity transfer vectors, and relation transfer vectors
		self.ent_embeddings = tf.get_variable(name = "ent_embeddings", shape = [config.entTotal, config.hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		self.rel_embeddings = tf.get_variable(name = "rel_embeddings", shape = [config.relTotal, config.hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		self.ent_transfer = tf.get_variable(name = "ent_transfer", shape = [config.entTotal, config.hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		self.rel_transfer = tf.get_variable(name = "rel_transfer", shape = [config.relTotal, config.hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))

	def loss_def(self):
		#Obtaining the initial configuration of the model
		config = self.get_config()
		#To get positive triples and negative triples for training
		#The shapes of pos_h, pos_t, pos_r are (batch_size, 1)
		#The shapes of neg_h, neg_t, neg_r are (batch_size, negative_ent + negative_rel)
		pos_h, pos_t, pos_r = self.get_positive_instance(in_batch = True)
		neg_h, neg_t, neg_r = self.get_negative_instance(in_batch = True)
		#Embedding entities and relations of triples, e.g. pos_h_e, pos_t_e and pos_r_e are embeddings for positive triples
		pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, pos_h)
		pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, pos_t)
		pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, pos_r)
		neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, neg_h)
		neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, neg_t)
		neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, neg_r)
		#Getting the required parameters to transfer entity embeddings, e.g. pos_h_t, pos_t_t and pos_r_t are transfer parameters for positive triples
		pos_h_t = tf.nn.embedding_lookup(self.ent_transfer, pos_h)
		pos_t_t = tf.nn.embedding_lookup(self.ent_transfer, pos_t)
		pos_r_t = tf.nn.embedding_lookup(self.rel_transfer, pos_r)
		neg_h_t = tf.nn.embedding_lookup(self.ent_transfer, neg_h)
		neg_t_t = tf.nn.embedding_lookup(self.ent_transfer, neg_t)
		neg_r_t = tf.nn.embedding_lookup(self.rel_transfer, neg_r)
		#Calculating score functions for all positive triples and negative triples
		p_h = self._transfer(pos_h_e, pos_h_t, pos_r_t)
		p_t = self._transfer(pos_t_e, pos_t_t, pos_r_t)
		p_r = pos_r_e
		n_h = self._transfer(neg_h_e, neg_h_t, neg_r_t)
		n_t = self._transfer(neg_t_e, neg_t_t, neg_r_t)
		n_r = neg_r_e
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
