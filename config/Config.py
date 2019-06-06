# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
import copy

import pdb

def to_var(x):
    return Variable(torch.from_numpy(x).cuda())


class Config(object):
    def __init__(self):
        base_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../release/Base.so")
        )
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        """argtypes"""
        """'sample"""
        self.lib.sampling.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        """'valid"""
        self.lib.getValidHeadBatch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.getValidTailBatch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.validHead.argtypes = [ctypes.c_void_p]
        self.lib.validTail.argtypes = [ctypes.c_void_p]
        """test link prediction"""
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
        self.lib.testHead.argtypes = [ctypes.c_void_p]
        self.lib.testTail.argtypes = [ctypes.c_void_p]
        """test triple classification"""
        self.lib.getValidBatch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.getTestBatch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.getBestThreshold.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.test_triple_classification.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.setHeadTailCrossSampling.argtypes = [
            ctypes.c_bool,
        ]
        """restype"""
        self.lib.getValidHit10.restype = ctypes.c_float
        self.lib.judgeHeadBatch.restype = ctypes.c_bool
        """set essential parameters"""
        self.in_path = "./"
        self.batch_size = 100
        self.bern = 0
        self.work_threads = 8
        self.hidden_size = 100
        self.negative_ent = 1
        self.negative_rel = 0
        self.ent_size = self.hidden_size
        self.rel_size = self.hidden_size
        self.head_batch = False
        self.margin = 1.0
        self.epsilon = 2.0
        self.adv_temperature = 1
        self.valid_steps = 5
        self.save_steps = 5
        self.opt_method = "SGD"
        self.optimizer = None
        self.lr_decay = 0
        self.weight_decay = 0
        self.lmbda = 0.0
        self.alpah = 0.001
        self.early_stopping_patience = 10
        self.nbatches = 100
        self.p_norm = 1
        self.self_adv = False
        self.test_link = True
        self.ent_double_embedding = False
        self.rel_double_embedding = False
        self.cross_sampling = False
        self.test_triple = True
        self.model = None
        self.trainModel = None
        self.testModel = None
        self.pretrain_model = None

    def init(self):
        self.lib.setInPath(
            ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2)
        )
        self.lib.setBern(self.bern)
        self.lib.setWorkThreads(self.work_threads)
        self.lib.randReset()
        self.lib.importTrainFiles()
        self.lib.importTestFiles()
        self.lib.importTypeFiles()
        self.relTotal = self.lib.getRelationTotal()
        self.entTotal = self.lib.getEntityTotal()
        self.trainTotal = self.lib.getTrainTotal()
        self.testTotal = self.lib.getTestTotal()
        self.validTotal = self.lib.getValidTotal()

        self.batch_size = int(self.trainTotal / self.nbatches)
        self.batch_seq_size = self.batch_size * (
            1 + self.negative_ent + self.negative_rel
        )
        self.batch_h = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_t = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_r = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_y = np.zeros(self.batch_seq_size, dtype=np.float32)
        self.batch_h_addr = self.batch_h.__array_interface__["data"][0]
        self.batch_t_addr = self.batch_t.__array_interface__["data"][0]
        self.batch_r_addr = self.batch_r.__array_interface__["data"][0]
        self.batch_y_addr = self.batch_y.__array_interface__["data"][0]

        self.valid_h = np.zeros(self.entTotal, dtype=np.int64)
        self.valid_t = np.zeros(self.entTotal, dtype=np.int64)
        self.valid_r = np.zeros(self.entTotal, dtype=np.int64)
        self.valid_h_addr = self.valid_h.__array_interface__["data"][0]
        self.valid_t_addr = self.valid_t.__array_interface__["data"][0]
        self.valid_r_addr = self.valid_r.__array_interface__["data"][0]

        self.test_h = np.zeros(self.entTotal, dtype=np.int64)
        self.test_t = np.zeros(self.entTotal, dtype=np.int64)
        self.test_r = np.zeros(self.entTotal, dtype=np.int64)
        self.test_h_addr = self.test_h.__array_interface__["data"][0]
        self.test_t_addr = self.test_t.__array_interface__["data"][0]
        self.test_r_addr = self.test_r.__array_interface__["data"][0]

        self.valid_pos_h = np.zeros(self.validTotal, dtype=np.int64)
        self.valid_pos_t = np.zeros(self.validTotal, dtype=np.int64)
        self.valid_pos_r = np.zeros(self.validTotal, dtype=np.int64)
        self.valid_pos_h_addr = self.valid_pos_h.__array_interface__["data"][0]
        self.valid_pos_t_addr = self.valid_pos_t.__array_interface__["data"][0]
        self.valid_pos_r_addr = self.valid_pos_r.__array_interface__["data"][0]
        self.valid_neg_h = np.zeros(self.validTotal, dtype=np.int64)
        self.valid_neg_t = np.zeros(self.validTotal, dtype=np.int64)
        self.valid_neg_r = np.zeros(self.validTotal, dtype=np.int64)
        self.valid_neg_h_addr = self.valid_neg_h.__array_interface__["data"][0]
        self.valid_neg_t_addr = self.valid_neg_t.__array_interface__["data"][0]
        self.valid_neg_r_addr = self.valid_neg_r.__array_interface__["data"][0]

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
        self.relThresh = np.zeros(self.relTotal, dtype=np.float32)
        self.relThresh_addr = self.relThresh.__array_interface__["data"][0]

    def set_test_link(self, test_link):
        self.test_link = test_link

    def set_test_triple(self, test_triple):
        self.test_triple = test_triple

    def set_margin(self, margin):
        self.margin = margin

    def set_in_path(self, in_path):
        self.in_path = in_path

    def set_nbatches(self, nbatches):
        self.nbatches = nbatches

    def set_p_norm(self, p_norm):
        self.p_norm = p_norm

    def set_valid_steps(self, valid_steps):
        self.valid_steps = valid_steps

    def set_save_steps(self, save_steps):
        self.save_steps = save_steps

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def set_result_dir(self, result_dir):
        self.result_dir = result_dir

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_lmbda(self, lmbda):
        self.lmbda = lmbda

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

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

    def set_train_times(self, train_times):
        self.train_times = train_times

    def set_work_threads(self, work_threads):
        self.work_threads = work_threads

    def set_ent_neg_rate(self, rate):
        self.negative_ent = rate

    def set_rel_neg_rate(self, rate):
        self.negative_rel = rate

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_self_adv(self, self_adv):
        self.self_adv = self_adv

    def set_self_adv_temperature(self, temperature):
        self.adv_temperature = temperature

    def set_cross_sampling(self, cross_sampling):
        self.cross_sampling = cross_sampling
        self.lib.setHeadTailCrossSampling(cross_sampling)
        # print(self.lib.judgeHeadBatch())

    def set_ent_double_embedding(self, ent_double_embedding):
        self.ent_double_embedding = ent_double_embedding

    def set_rel_double_embedding(self, rel_double_embedding):
        self.rel_double_embedding = rel_double_embedding

    def set_early_stopping_patience(self, early_stopping_patience):
        self.early_stopping_patience = early_stopping_patience

    def set_pretrain_model(self, pretrain_model):
        self.pretrain_model = pretrain_model

    def get_parameters(self, param_dict, mode="numpy"):
        for param in param_dict:
            param_dict[param] = param_dict[param].cpu()
        res = {}
        for param in param_dict:
            if mode == "numpy":
                res[param] = param_dict[param].numpy()
            elif mode == "list":
                res[param] = param_dict[param].numpy().tolist()
            else:
                res[param] = param_dict[param]
        return res

    def save_embedding_matrix(self, best_model):
        path = os.path.join(self.result_dir, self.model.__name__ + ".json")
        f = open(path, "w")
        f.write(json.dumps(self.get_parameters(best_model, "list")))
        f.close()

    def set_train_model(self, model):
        print("Initializing training model...")
        self.model = model
        self.trainModel = self.model(config=self)
        self.trainModel.cuda()
        if self.optimizer != None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(
                self.trainModel.parameters(),
                lr=self.alpha,
                lr_decay=self.lr_decay,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(
                self.trainModel.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.trainModel.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = optim.SGD(
                self.trainModel.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        print("Finish initializing")

    def set_test_model(self, model, path=None):
        print("Initializing test model...")
        self.model = model
        self.testModel = self.model(config=self)
        if path == None:
            path = os.path.join(self.result_dir, self.model.__name__ + ".ckpt")
        self.testModel.load_state_dict(torch.load(path))
        self.testModel.cuda()
        self.testModel.eval()
        print("Finish initializing")

    def sampling(self):
        self.head_batch = self.lib.judgeHeadBatch()
        self.lib.sampling(
            self.batch_h_addr,
            self.batch_t_addr,
            self.batch_r_addr,
            self.batch_y_addr,
            self.batch_size,
            self.negative_ent,
            self.negative_rel,
        )

    def save_checkpoint(self, model, epoch):
        path = os.path.join(
            self.checkpoint_dir, self.model.__name__ + "-" + str(epoch) + ".ckpt"
        )
        torch.save(model, path)

    def save_best_checkpoint(self, best_model):
        path = os.path.join(self.result_dir, self.model.__name__ + ".ckpt")
        torch.save(best_model, path)

    def train_one_step(self):
        self.trainModel.batch_h = to_var(self.batch_h)
        self.trainModel.batch_t = to_var(self.batch_t)
        self.trainModel.batch_r = to_var(self.batch_r)
        self.trainModel.batch_y = to_var(self.batch_y)
        self.optimizer.zero_grad()
        loss = self.trainModel(self.head_batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test_one_step(self, model, test_h, test_t, test_r):
        model.batch_h = to_var(test_h)
        model.batch_t = to_var(test_t)
        model.batch_r = to_var(test_r)
        return model.predict()

    def valid(self, model):
        self.lib.validInit()
        for i in range(self.validTotal):
            sys.stdout.write("%d\r" % (i))
            sys.stdout.flush()
            self.lib.getValidHeadBatch(
                self.valid_h_addr, self.valid_t_addr, self.valid_r_addr
            )
            res = self.test_one_step(model, self.valid_h, self.valid_t, self.valid_r)

            self.lib.validHead(res.__array_interface__["data"][0])

            self.lib.getValidTailBatch(
                self.valid_h_addr, self.valid_t_addr, self.valid_r_addr
            )
            res = self.test_one_step(model, self.valid_h, self.valid_t, self.valid_r)
            self.lib.validTail(res.__array_interface__["data"][0])
        return self.lib.getValidHit10()

    def train(self):
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        best_epoch = 0
        best_hit10 = 0.0
        best_model = None
        bad_counts = 0
        for epoch in range(self.train_times):
            res = 0.0
            for batch in range(self.nbatches):
                self.sampling()
                loss = self.train_one_step()
                res += loss
            print("Epoch %d | loss: %f" % (epoch, res))
            if (epoch + 1) % self.save_steps == 0:
                print("Epoch %d has finished, saving..." % (epoch))
                self.save_checkpoint(self.trainModel.state_dict(), epoch)
            if (epoch + 1) % self.valid_steps == 0:
                print("Epoch %d has finished, validating..." % (epoch))
                hit10 = self.valid(self.trainModel)
                if hit10 > best_hit10:
                    best_hit10 = hit10
                    best_epoch = epoch
                    best_model = copy.deepcopy(self.trainModel.state_dict())
                    print("Best model | hit@10 of valid set is %f" % (best_hit10))
                    bad_counts = 0
                else:
                    print(
                        "Hit@10 of valid set is %f | bad count is %d"
                        % (hit10, bad_counts)
                    )
                    bad_counts += 1
                if bad_counts == self.early_stopping_patience:
                    print("Early stopping at epoch %d" % (epoch))
                    break
        if best_model == None:
            best_model = self.trainModel.state_dict()
            best_epoch = self.train_times - 1
            best_hit10 = self.valid(self.trainModel)
        print("Best epoch is %d | hit@10 of valid set is %f" % (best_epoch, best_hit10))
        print("Store checkpoint of best result at epoch %d..." % (best_epoch))
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)
        self.save_best_checkpoint(best_model)
        self.save_embedding_matrix(best_model)
        print("Finish storing")
        print("Testing...")
        self.set_test_model(self.model)
        self.test()
        print("Finish test")
        return best_model

    def link_prediction(self):
        print("The total of test triple is %d" % (self.testTotal))
        for i in range(self.testTotal):
            sys.stdout.write("%d\r" % (i))
            sys.stdout.flush()
            self.lib.getHeadBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
            res = self.test_one_step(
                self.testModel, self.test_h, self.test_t, self.test_r
            )
            self.lib.testHead(res.__array_interface__["data"][0])

            self.lib.getTailBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
            res = self.test_one_step(
                self.testModel, self.test_h, self.test_t, self.test_r
            )
            self.lib.testTail(res.__array_interface__["data"][0])
        self.lib.test_link_prediction()

    def triple_classification(self):
        self.lib.getValidBatch(
            self.valid_pos_h_addr,
            self.valid_pos_t_addr,
            self.valid_pos_r_addr,
            self.valid_neg_h_addr,
            self.valid_neg_t_addr,
            self.valid_neg_r_addr,
        )
        res_pos = self.test_one_step(
            self.testModel, self.valid_pos_h, self.valid_pos_t, self.valid_pos_r
        )
        res_neg = self.test_one_step(
            self.testModel, self.valid_neg_h, self.valid_neg_t, self.valid_neg_r
        )
        self.lib.getBestThreshold(
            self.relThresh_addr,
            res_pos.__array_interface__["data"][0],
            res_neg.__array_interface__["data"][0],
        )

        self.lib.getTestBatch(
            self.test_pos_h_addr,
            self.test_pos_t_addr,
            self.test_pos_r_addr,
            self.test_neg_h_addr,
            self.test_neg_t_addr,
            self.test_neg_r_addr,
        )
        res_pos = self.test_one_step(
            self.testModel, self.test_pos_h, self.test_pos_t, self.test_pos_r
        )
        res_neg = self.test_one_step(
            self.testModel, self.test_neg_h, self.test_neg_t, self.test_neg_r
        )
        self.lib.test_triple_classification(
            self.relThresh_addr,
            res_pos.__array_interface__["data"][0],
            res_neg.__array_interface__["data"][0],
        )

    def test(self):
        if self.test_link:
            self.link_prediction()
        if self.test_triple:
            self.triple_classification()
