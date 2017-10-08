#coding:utf-8
import numpy as np
import tensorflow as tf
import os
import time
import datetime
import ctypes

class Config(object):

	def __init__(self):
		self.lib = ctypes.cdll.LoadLibrary("./release/Base.so")
		self.lib.sampling.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
		self.in_path = "./"
		self.out_path = "./"
		self.bern = 1
		self.hidden_size = 100
		self.ent_size = self.hidden_size
		self.rel_size = self.hidden_size
		self.train_times = 1000
		self.margin = 1.0
		self.nbatches = 100
		self.negative_ent = 1
		self.negative_rel = 0
		self.workThreads = 1
		self.alpha = 0.001
		self.lmbda = 0.000
		self.log_on = 1
		self.exportName = None
		self.importName = None
		self.export_steps = 1
		self.optimizer = "SGD"

	def init(self):
		self.lib.setInPath(ctypes.create_string_buffer(self.in_path, len(self.in_path) * 2))
		self.lib.setOutPath(ctypes.create_string_buffer(self.out_path, len(self.out_path) * 2))
		self.lib.setBern(self.bern)
		self.lib.setWorkThreads(self.workThreads)
		self.lib.randReset()
		self.lib.importTrainFiles()
		self.relTotal = self.lib.getRelationTotal()
		self.entTotal = self.lib.getEntityTotal()
		self.tripleTotal = self.lib.getTripleTotal()
		self.batch_size = self.lib.getTripleTotal() / self.nbatches
		self.batch_seq_size = self.batch_size * (1 + self.negative_ent + self.negative_rel)
		self.batch_h = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
		self.batch_t = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
		self.batch_r = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
		self.batch_y = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.float32)
		self.batch_h_addr = self.batch_h.__array_interface__['data'][0]
		self.batch_t_addr = self.batch_t.__array_interface__['data'][0]
		self.batch_r_addr = self.batch_r.__array_interface__['data'][0]
		self.batch_y_addr = self.batch_y.__array_interface__['data'][0]

	def set_lmbda(self, lmbda):
		self.lmbda = lmbda

	def set_optimizer(self, optimizer):
		self.optimizer = optimizer

	def set_log_on(self, flag):
		self.log_on = flag

	def set_alpha(self, alpha):
		self.alpha = alpha

	def set_in_path(self, path):
		self.in_path = path

	def set_out_path(self, path):
		self.out_path = path

	def set_bern(self, bern):
		self.bern = bern

	def set_dimension(self, dim):
		self.hidden_size = dim
		self.ent_size = dim
		self.rel_size = dim

	def set_ent_dimension(self, dim):
		self.ent_size = dim

	def set_rel_dimension(self, dim):
		self.rel_size = dim

	def set_train_times(self, times):
		self.train_times = times

	def set_nbatches(self, nbatches):
		self.nbatches = nbatches

	def set_margin(self, margin):
		self.margin = margin

	def set_work_threads(self, threads):
		self.workThreads = threads

	def set_ent_neg_rate(self, rate):
		self.negative_ent = rate

	def set_rel_neg_rate(self, rate):
		self.negative_rel = rate

	def set_import_files(self, path):
		self.importName = path

	def set_export_files(self, path):
		self.exportName = path

	def set_export_steps(self, steps):
		self.export_steps = steps

	def sampling(self):
		self.lib.sampling(self.batch_h_addr, self.batch_t_addr, self.batch_r_addr, self.batch_y_addr, self.batch_size, self.negative_ent, self.negative_rel)

	def save_tensorflow(self):
		with self.graph.as_default():
			with self.sess.as_default():
				self.saver.save(self.sess, self.exportName)

	def restore_tensorflow(self):
		with self.graph.as_default():
			with self.sess.as_default():
				self.saver.restore(self.sess, self.importName)


	def export_variables(self, path):
		with self.graph.as_default():
			with self.sess.as_default():
				self.saver.save(self.sess, path)

	def import_variables(self, path):
		with self.graph.as_default():
			with self.sess.as_default():
				self.saver.restore(self.sess, path)

	def set_model(self, model):
		self.model = model
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.sess = tf.Session()
			with self.sess.as_default():
				initializer = tf.contrib.layers.xavier_initializer(uniform = True)
				with tf.variable_scope("model", reuse=None, initializer = initializer):
					self.trainModel = self.model(config = self)
					if self.optimizer == "Adagrad" or self.optimizer == "adagrad":
						optimizer = tf.train.AdagradOptimizer(learning_rate = self.alpha, initial_accumulator_value=1e-8)
					elif self.optimizer == "Adadelta" or self.optimizer == "adadelta":
						optimizer = tf.train.AdadeltaOptimizer(self.alpha)
					elif self.optimizer == "Adam" or self.optimizer == "adam":
						optimizer = tf.train.AdamOptimizer(self.alpha)
					else:
						optimizer = tf.train.GradientDescentOptimizer(self.alpha)
					grads_and_vars = optimizer.compute_gradients(self.trainModel.loss)
					self.train_op = optimizer.apply_gradients(grads_and_vars)
				self.saver = tf.train.Saver()



	def train_step(self, batch_h, batch_t, batch_r, batch_y):
		feed_dict = {
			self.trainModel.batch_h: batch_h,
			self.trainModel.batch_t: batch_t,
			self.trainModel.batch_r: batch_r,
			self.trainModel.batch_y: batch_y
		}
		_, loss = self.sess.run([self.train_op, self.trainModel.loss], feed_dict)
	 	return loss

	def run(self):
		with self.graph.as_default():
			with self.sess.as_default():
				self.sess.run(tf.initialize_all_variables())
				if self.importName != None:
					self.restore_tensorflow()
				for times in range(self.train_times):
					res = 0.0
					for batch in range(self.nbatches):
						self.sampling()
						res += self.train_step(self.batch_h, self.batch_t, self.batch_r, self.batch_y)
					if self.log_on:
						print times
						print res
					if self.exportName != None and times % self.export_steps == 0:
						self.save_tensorflow()

