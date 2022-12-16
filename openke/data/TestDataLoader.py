# coding:utf-8
import os
import ctypes
import numpy as np

class TestDataSampler(object):

	def __init__(self, data_total, data_sampler):
		self.data_total = data_total
		self.data_sampler = data_sampler
		self.total = 0

	def __iter__(self):
		return self

	def __next__(self):
		self.total += 1 
		if self.total > self.data_total:
			raise StopIteration()
		return self.data_sampler()

	def __len__(self):
		return self.data_total

class TestDataLoader(object):

	def __init__(self, in_path = "./", sampling_mode = 'link', type_constrain = True):
		base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
		self.lib = ctypes.cdll.LoadLibrary(base_file)
		"""for link prediction"""
		self.lib.getHeadBatch.argtypes = [
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
		]
		self.lib.getTailBatch.argtypes = [
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
		]
		"""for triple classification"""
		self.lib.getTestBatch.argtypes = [
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
		]
		"""set essential parameters"""
		self.in_path = in_path
		self.sampling_mode = sampling_mode
		self.type_constrain = type_constrain
		self.read()

	def read(self):
		self.lib.setInPath(ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2))
		self.lib.randReset()
		self.lib.importTestFiles()

		if self.type_constrain:
			self.lib.importTypeFiles()

		self.relTotal = self.lib.getRelationTotal()
		self.entTotal = self.lib.getEntityTotal()
		self.testTotal = self.lib.getTestTotal()

		self.test_h = np.zeros(self.entTotal, dtype=np.int64)
		self.test_t = np.zeros(self.entTotal, dtype=np.int64)
		self.test_r = np.zeros(self.entTotal, dtype=np.int64)
		self.test_h_addr = self.test_h.__array_interface__["data"][0]
		self.test_t_addr = self.test_t.__array_interface__["data"][0]
		self.test_r_addr = self.test_r.__array_interface__["data"][0]

		self.test_pos_h = np.zeros(self.testTotal, dtype=np.int64)
		self.test_pos_t = np.zeros(self.testTotal, dtype=np.int64)
		self.test_pos_r = np.zeros(self.testTotal, dtype=np.int64)
		self.test_pos_h_addr = self.test_pos_h.__array_interface__["data"][0]
		self.test_pos_t_addr = self.test_pos_t.__array_interface__["data"][0]
		self.test_pos_r_addr = self.test_pos_r.__array_interface__["data"][0]
		self.test_neg_h = np.zeros(self.testTotal, dtype=np.int64)
		self.test_neg_t = np.zeros(self.testTotal, dtype=np.int64)
		self.test_neg_r = np.zeros(self.testTotal, dtype=np.int64)
		self.test_neg_h_addr = self.test_neg_h.__array_interface__["data"][0]
		self.test_neg_t_addr = self.test_neg_t.__array_interface__["data"][0]
		self.test_neg_r_addr = self.test_neg_r.__array_interface__["data"][0]

	def sampling_lp(self):
		res = []
		self.lib.getHeadBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
		res.append({
			"batch_h": self.test_h.copy(), 
			"batch_t": self.test_t[:1].copy(), 
			"batch_r": self.test_r[:1].copy(),
			"mode": "head_batch"
		})
		self.lib.getTailBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
		res.append({
			"batch_h": self.test_h[:1],
			"batch_t": self.test_t,
			"batch_r": self.test_r[:1],
			"mode": "tail_batch"
		})
		return res

	def sampling_tc(self):
		self.lib.getTestBatch(
			self.test_pos_h_addr,
			self.test_pos_t_addr,
			self.test_pos_r_addr,
			self.test_neg_h_addr,
			self.test_neg_t_addr,
			self.test_neg_r_addr,
		)
		return [ 
			{
				'batch_h': self.test_pos_h,
				'batch_t': self.test_pos_t,
				'batch_r': self.test_pos_r ,
				"mode": "normal"
			}, 
			{
				'batch_h': self.test_neg_h,
				'batch_t': self.test_neg_t,
				'batch_r': self.test_neg_r,
				"mode": "normal"
			}
		]

	"""interfaces to get essential parameters"""

	def get_ent_tot(self):
		return self.entTotal

	def get_rel_tot(self):
		return self.relTotal

	def get_triple_tot(self):
		return self.testTotal

	def set_sampling_mode(self, sampling_mode):
		self.sampling_mode = sampling_mode

	def __len__(self):
		return self.testTotal

	def __iter__(self):
		if self.sampling_mode == "link":
			self.lib.initTest()
			return TestDataSampler(self.testTotal, self.sampling_lp)
		else:
			self.lib.initTest()
			return TestDataSampler(1, self.sampling_tc)