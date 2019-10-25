#coding:utf-8
import numpy as np
import tensorflow as tf
from .Model import Model

if tf.__version__ > '0.12.1':
	matmul_func = tf.matmul
else:
	matmul_func = tf.batch_matmul

class TransR(Model):
	r'''
	TransR first projects entities from entity space to corresponding relation space 
	and then builds translations between projected entities. 
	'''
	def _transfer(self, transfer_matrix, embeddings):
		return matmul_func(embeddings, transfer_matrix)

	def _calc(self, h, t, r):
		h = tf.nn.l2_normalize(h, -1)
		t = tf.nn.l2_normalize(t, -1)
		r = tf.nn.l2_normalize(r, -1)
		return abs(h + r - t)

	def embedding_def(self):
		#Obtaining the initial configuration of the model
		config = self.get_config()
		#Defining required parameters of the model, including embeddings of entities and relations, and mapping matrices
		self.ent_embeddings = tf.keras.layers.Embedding(config.entTotal, config.ent_size, name="ent_embeddings")
		self.rel_embeddings = tf.keras.layers.Embedding(config.relTotal, config.rel_size, name="rel_embeddings")
		self.transfer_matrix = tf.keras.layers.Embedding(config.relTotal, config.ent_size * config.rel_size, name="transfer_matrix")
		self.parameter_lists = {"ent_embeddings":self.ent_embeddings, \
								"rel_embeddings":self.rel_embeddings, \
								"transfer_matrix":self.transfer_matrix}

	def embeddings_look_up_htr(self, h, t, r):
		return [self.ent_embeddings(h), self.ent_embeddings(t), self.rel_embeddings(r)]

	def loss_def(self):
		#Obtaining the initial configuration of the model
		config = self.get_config()
		#To get positive triples and negative triples for training
		#The shapes of pos_h, pos_t, pos_r are (batch_size, 1)
		#The shapes of neg_h, neg_t, neg_r are (batch_size, negative_ent + negative_rel)
		pos_h, pos_t, pos_r = self.get_positive_instance(in_batch = True)
		neg_h, neg_t, neg_r = self.get_negative_instance(in_batch = True)
		#Embedding entities and relations of triples, e.g. pos_h_e, pos_t_e and pos_r_e are embeddings for positive triples
		pos_h_e, pos_t_e, pos_r_e = self.embeddings_look_up_htr(pos_h, pos_t, pos_r)
		neg_h_e, neg_t_e, neg_r_e = self.embeddings_look_up_htr(neg_h, neg_t, neg_r)
		#Getting the required mapping matrices
		pos_matrix = tf.reshape(self.transfer_matrix(pos_r), [-1, config.ent_size, config.rel_size])
		#Calculating score functions for all positive triples and negative triples
		p_h = self._transfer(pos_matrix, pos_h_e)
		p_t = self._transfer(pos_matrix, pos_t_e)
		p_r = pos_r_e
		if config.negative_rel == 0:
			n_h = self._transfer(pos_matrix, neg_h_e)
			n_t = self._transfer(pos_matrix, neg_t_e)
			n_r = neg_r_e
		else:
			neg_matrix = tf.reshape(self.transfer_matrix(neg_r), [-1, config.ent_size, config.rel_size])
			n_h = self._transfer(neg_matrix, neg_h_e)
			n_t = self._transfer(neg_matrix, neg_t_e)
			n_r = neg_r_e
		#The shape of _p_score is (batch_size, 1, hidden_size)
		#The shape of _n_score is (batch_size, negative_ent + negative_rel, hidden_size)
		_p_score = self._calc(p_h, p_t, p_r)
		_n_score = self._calc(n_h, n_t, n_r)
		#The shape of p_score is (batch_size, 1, 1)
		#The shape of n_score is (batch_size, negative_ent + negative_rel, 1)
		p_score =  tf.reduce_sum(_p_score, -1, keep_dims = True)
		n_score =  tf.reduce_sum(_n_score, -1, keep_dims = True)
		#Calculating loss to get what the framework will optimize
		self.loss = tf.reduce_mean(tf.maximum(p_score - n_score + config.margin, 0))

	def predict_def(self):
		config = self.get_config()
		predict_h, predict_t, predict_r = self.get_predict_instance()
		predict_h_e = tf.reshape(self.ent_embeddings(predict_h), [1, -1, config.ent_size])
		predict_t_e = tf.reshape(self.ent_embeddings(predict_t), [1, -1, config.ent_size])
		predict_r_e = tf.reshape(self.rel_embeddings(predict_r), [1, -1, config.rel_size])
		predict_matrix = tf.reshape(self.transfer_matrix(predict_r[0]), [1, config.ent_size, config.rel_size])
		h_e = tf.reshape(self._transfer(predict_matrix, predict_h_e), [-1, config.rel_size])
		t_e = tf.reshape(self._transfer(predict_matrix, predict_t_e), [-1, config.rel_size])
		r_e = predict_r_e
		self.predict = tf.reduce_sum(self._calc(h_e, t_e, r_e), -1, keep_dims = True)
