import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model


class ComplEx(Model):
    def __init__(self, config):
        super(ComplEx, self).__init__(config)
        self.ent_re_embeddings = nn.Embedding(
            self.config.entTotal, self.config.hidden_size
        )
        self.ent_im_embeddings = nn.Embedding(
            self.config.entTotal, self.config.hidden_size
        )
        self.rel_re_embeddings = nn.Embedding(
            self.config.relTotal, self.config.hidden_size
        )
        self.rel_im_embeddings = nn.Embedding(
            self.config.relTotal, self.config.hidden_size
        )
        self.criterion = nn.Softplus()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform(self.ent_re_embeddings.weight.data)
        nn.init.xavier_uniform(self.ent_im_embeddings.weight.data)
        nn.init.xavier_uniform(self.rel_re_embeddings.weight.data)
        nn.init.xavier_uniform(self.rel_im_embeddings.weight.data)

    def _calc(self, h_re, h_im, t_re, t_im, r_re, r_im):
        return -torch.sum(
            h_re * t_re * r_re
            + h_im * t_im * r_re
            + h_re * t_im * r_im
            - h_im * t_re * r_im,
            -1,
        )

    def loss(self, score, regul):
        return (
            torch.mean(self.criterion(score * self.batch_y)) + self.config.lmbda * regul
        )

    def forward(self):
        h_re = self.ent_re_embeddings(self.batch_h)
        h_im = self.ent_im_embeddings(self.batch_h)
        t_re = self.ent_re_embeddings(self.batch_t)
        t_im = self.ent_im_embeddings(self.batch_t)
        r_re = self.rel_re_embeddings(self.batch_r)
        r_im = self.rel_im_embeddings(self.batch_r)
        score = self._calc(h_re, h_im, t_re, t_im, r_re, r_im)
        regul = (
            torch.mean(h_re ** 2)
            + torch.mean(h_im ** 2)
            + torch.mean(t_re ** 2)
            + torch.mean(t_im ** 2)
            + torch.mean(r_re ** 2)
            + torch.mean(r_im ** 2)
        )
        return self.loss(score, regul)

    def predict(self):
        h_re = self.ent_re_embeddings(self.batch_h)
        h_im = self.ent_im_embeddings(self.batch_h)
        t_re = self.ent_re_embeddings(self.batch_t)
        t_im = self.ent_im_embeddings(self.batch_t)
        r_re = self.rel_re_embeddings(self.batch_r)
        r_im = self.rel_im_embeddings(self.batch_r)
        score = self._calc(h_re, h_im, t_re, t_im, r_re, r_im)
        return score.cpu().data.numpy()
