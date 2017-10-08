#coding:utf-8
import numpy as np
import tensorflow as tf
from Model import *

class ComplEx(Model):

	def embedding_def(self):
		config = self.get_config()
		self.ent1_embeddings = tf.Variable(tf.truncated_normal(([config.entTotal, config.hidden_size])), name = "ent1_embedding")
		self.rel1_embeddings = tf.Variable(tf.truncated_normal(([config.relTotal, config.hidden_size])), name = "rel1_embedding")
		self.ent2_embeddings = tf.Variable(tf.truncated_normal(([config.entTotal, config.hidden_size])), name = "ent2_embedding")
		self.rel2_embeddings = tf.Variable(tf.truncated_normal(([config.relTotal, config.hidden_size])), name = "rel2_embedding")

	def loss_def(self):
		#Obtaining the initial configuration of the model
		config = self.get_config()
		#To get positive triples and negative triples for training
		#To get labels for the triples, positive triples as 1 and negative triples as -1
		#The shapes of h, t, r, y are (batch_size, 1 + negative_ent + negative_rel)
		h, t, r = self.get_all_instance()
		y = self.get_all_labels()
		#Embedding entities and relations of triples
		e1_h = tf.nn.embedding_lookup(self.ent1_embeddings, h)
		e2_h = tf.nn.embedding_lookup(self.ent2_embeddings, h)
		e1_t = tf.nn.embedding_lookup(self.ent1_embeddings, t)
		e2_t = tf.nn.embedding_lookup(self.ent2_embeddings, t)
		r1 = tf.nn.embedding_lookup(self.rel1_embeddings, r)
		r2 = tf.nn.embedding_lookup(self.rel2_embeddings, r)
		#Calculating score functions for all positive triples and negative triples
		res = tf.reduce_sum(e1_h * e1_t * r1 + e2_h * e2_t * r1 + e1_h * e2_t * r2 - e2_h * e1_t * r2, 1, keep_dims = False)
		loss_func = tf.reduce_mean(tf.nn.softplus(- y * res), 0, keep_dims = False)
		regul_func = tf.reduce_mean(e1_h ** 2) + tf.reduce_mean(e1_t ** 2) + tf.reduce_mean(e2_h ** 2) + tf.reduce_mean(e2_t ** 2) + tf.reduce_mean(r1 ** 2) + tf.reduce_mean(r2 ** 2)
		#Calculating loss to get what the framework will optimize
		self.loss =  loss_func + config.lmbda * regul_func
