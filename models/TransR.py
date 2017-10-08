#coding:utf-8
import numpy as np
import tensorflow as tf
from Model import *

class TransR(Model):

	def _transfer(self, transfer_matrix, embeddings):
		return tf.batch_matmul(transfer_matrix, embeddings)

	def _calc(self, h, t, r):
		return abs(h + r - t)

	def embedding_def(self):
		#Obtaining the initial configuration of the model
		config = self.get_config()
		#Defining required parameters of the model, including embeddings of entities and relations, and mapping matrices
		self.ent_embeddings = tf.get_variable(name = "ent_embeddings", shape = [config.entTotal, config.ent_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		self.rel_embeddings = tf.get_variable(name = "rel_embeddings", shape = [config.relTotal, config.rel_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		self.transfer_matrix = tf.get_variable(name = "rel_matrix", shape = [config.relTotal, config.ent_size * config.rel_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))

	def loss_def(self):
		#Obtaining the initial configuration of the model
		config = self.get_config()
		#To get positive triples and negative triples for training
		#The shapes of pos_h, pos_t, pos_r are (batch_size, 1)
		#The shapes of neg_h, neg_t, neg_r are (batch_size, negative_ent + negative_rel)
		pos_h, pos_t, pos_r = self.get_positive_instance(in_batch = True)
		neg_h, neg_t, neg_r = self.get_negative_instance(in_batch = True)
		#Embedding entities and relations of triples, e.g. pos_h_e, pos_t_e and pos_r_e are embeddings for positive triples
		pos_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, pos_h), [-1, config.ent_size, 1])
		pos_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, pos_t), [-1, config.ent_size, 1])
		pos_r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings, pos_r), [-1, config.rel_size])
		neg_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, neg_h), [-1, config.ent_size, 1])
		neg_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, neg_t), [-1, config.ent_size, 1])
		neg_r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings, neg_r), [-1, config.rel_size])
		#Getting the required mapping matrices
		pos_matrix = tf.reshape(tf.nn.embedding_lookup(self.transfer_matrix, pos_r), [-1, config.rel_size, config.ent_size])
		neg_matrix = tf.reshape(tf.nn.embedding_lookup(self.transfer_matrix, neg_r), [-1, config.rel_size, config.ent_size])
		#Calculating score functions for all positive triples and negative triples
		p_h = tf.reshape(self._transfer(pos_matrix, pos_h_e), [-1, config.rel_size])
		p_t = tf.reshape(self._transfer(pos_matrix, pos_t_e), [-1, config.rel_size])
		p_r = pos_r_e
		n_h = tf.reshape(self._transfer(neg_matrix, neg_h_e), [-1, config.rel_size])
		n_t = tf.reshape(self._transfer(neg_matrix, neg_t_e), [-1, config.rel_size])
		n_r = neg_r_e
		#The shape of _p_score is (batch_size, 1, hidden_size)
		#The shape of _n_score is (batch_size, negative_ent + negative_rel, hidden_size)
		_p_score = self._calc(p_h, p_t, p_r)
		_p_score = tf.reshape(_p_score, [-1, 1, config.rel_size])
		_n_score = self._calc(n_h, n_t, n_r)
		_n_score = tf.reshape(_n_score, [-1, config.negative_ent + config.negative_rel, config.rel_size])
		#The shape of p_score is (batch_size, 1)
		#The shape of n_score is (batch_size, 1)
		p_score =  tf.reduce_sum(tf.reduce_mean(_p_score, 1, keep_dims = False), 1, keep_dims = True)
		n_score =  tf.reduce_sum(tf.reduce_mean(_n_score, 1, keep_dims = False), 1, keep_dims = True)
		#Calculating loss to get what the framework will optimize
		self.loss = tf.reduce_sum(tf.maximum(p_score - n_score + config.margin, 0))
