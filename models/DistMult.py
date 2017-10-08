#coding:utf-8
import numpy as np
import tensorflow as tf
from Model import *

class DistMult(Model):

	def embedding_def(self):
		config = self.get_config()
		self.ent_embeddings = tf.Variable(tf.truncated_normal(([config.entTotal, config.hidden_size])), name = "ent_embedding")
		self.rel_embeddings = tf.Variable(tf.truncated_normal(([config.relTotal, config.hidden_size])), name = "rel_embedding")

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
		res = tf.reduce_sum(e_h * e_t * e_r, 1, keep_dims = False)
		loss_func = tf.reduce_mean(tf.nn.softplus(- y * res))
		regul_func = tf.reduce_mean(e_h ** 2) + tf.reduce_mean(e_t ** 2) + tf.reduce_mean(e_r ** 2)
		#Calculating loss to get what the framework will optimize
		self.loss =  loss_func + config.lmbda * regul_func
