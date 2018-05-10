#coding:utf-8
import numpy as np
import tensorflow as tf
from Model import *

class Analogy(Model):

    def _calc(self, e_re_h, e_im_h, e_h, e_re_t, e_im_t, e_t, r_re, r_im, r):
        return tf.reduce_sum(r_re * e_re_h * e_re_t + r_re * e_im_h * e_im_t + r_im * e_re_h * e_im_t - r_im * e_im_h * e_re_t, axis=1, keepdims=False) + tf.reduce_sum(e_h * e_t * r, axis=1, keepdims=False)

    def embedding_def(self):
        #Obtaining the initial configuration of the model
        config = self.get_config()
        #Defining required parameters of the model, including embeddings of entities and relations, entity transfer vectors, and relation transfer vectors
        self.ent_embeddings = tf.get_variable(name = "ent_embeddings", shape = [config.entTotal, config.hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform=True))
        self.rel_embeddings = tf.get_variable(name = "rel_embeddings", shape = [config.relTotal, config.hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform=True))
        self.ent_re = tf.get_variable(name = "ent_re", shape = [config.entTotal, config.hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform=True))
        self.rel_re = tf.get_variable(name = "rel_re", shape = [config.relTotal, config.hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform=True))
        self.ent_im = tf.get_variable(name = "ent_im", shape = [config.entTotal, config.hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform=True))
        self.rel_im = tf.get_variable(name = "rel_im", shape = [config.relTotal, config.hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform=True))
        self.parameter_lists = {"ent_embeddings":self.ent_embeddings, \
                                "rel_embeddings":self.rel_embeddings, \
                                "ent_re":self.ent_re, \
                                "rel_re":self.rel_re, \
                                "ent_im":self.ent_im, \
                                "rel_im":self.rel_im}

    def loss_def(self):
        #Obtaining the initial configuration of the model
        config = self.get_config()
        # get triples and labels
        id_h, id_t, id_r = self.get_all_instance()
        y = self.get_all_labels()
        # get the embeddings
        e_re_h = tf.nn.embedding_lookup(self.ent_re, id_h)
        e_im_h = tf.nn.embedding_lookup(self.ent_im, id_h)
        e_h    = tf.nn.embedding_lookup(self.ent_embeddings, id_h)
        e_re_t = tf.nn.embedding_lookup(self.ent_re, id_t)
        e_im_t = tf.nn.embedding_lookup(self.ent_im, id_t)
        e_t    = tf.nn.embedding_lookup(self.ent_embeddings, id_t)
        r_re   = tf.nn.embedding_lookup(self.rel_re, id_r)
        r_im   = tf.nn.embedding_lookup(self.rel_im, id_r)
        r      = tf.nn.embedding_lookup(self.rel_embeddings, id_r)
        # calculating loss with regularization
        res = self._calc(e_re_h, e_im_h, e_h, e_re_t, e_im_t, e_t, r_re, r_im, r)
        loss = tf.reduce_mean(tf.nn.softplus(- y * res))
        regul = tf.reduce_mean(e_re_h**2)+tf.reduce_mean(e_im_h**2)*tf.reduce_mean(e_h**2)+tf.reduce_mean(e_re_t**2)+tf.reduce_mean(e_im_t**2)+tf.reduce_mean(e_t**2)+tf.reduce_mean(r_re**2)+tf.reduce_mean(r_im**2)+tf.reduce_mean(r**2)
        self.loss = loss + self.config.lmbda * regul

    def predict_def(self):
        config = self.get_config()
        id_h, id_t, id_r = self.get_predict_instance()
        e_re_h = tf.nn.embedding_lookup(self.ent_re, id_h)
        e_im_h = tf.nn.embedding_lookup(self.ent_im, id_h)
        e_h    = tf.nn.embedding_lookup(self.ent_embeddings, id_h)
        e_re_t = tf.nn.embedding_lookup(self.ent_re, id_t)
        e_im_t = tf.nn.embedding_lookup(self.ent_im, id_t)
        e_t    = tf.nn.embedding_lookup(self.ent_embeddings, id_t)
        r_re   = tf.nn.embedding_lookup(self.rel_re, id_r)
        r_im   = tf.nn.embedding_lookup(self.rel_im, id_r)
        r      = tf.nn.embedding_lookup(self.rel_embeddings, id_r)
        self.predict = -self._calc(e_re_h, e_im_h, e_h, e_re_t, e_im_t, e_t, r_re, r_im, r)

