import torch
import torch.nn as nn
from .Model import Model


class SimplE(Model):
    def __init__(self, config):
        super(SimplE, self).__init__(config)
        self.ent_embeddings = nn.Embedding(
            self.config.entTotal, self.config.hidden_size)
        self.rel_embeddings = nn.Embedding(
            self.config.relTotal, self.config.hidden_size)
        self.rel_inv_embeddings = nn.Embedding(
            self.config.relTotal, self.config.hidden_size)
        self.criterion = nn.Softplus()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform(self.rel_inv_embeddings.weight.data)

    def _calc_avg(self, h, t, r, r_inv):
        return (- torch.sum(h * r * t, -1) - torch.sum(h * r_inv * t, -1))/2

    def _calc_ingr(self, h, r, t):
        return -torch.sum(h * r * t, -1)

    def loss(self, score, regul):
        return torch.mean(self.criterion(score * self.batch_y)) + self.config.lmbda * regul

    def forward(self):
        h = self.ent_embeddings(self.batch_h)
        t = self.ent_embeddings(self.batch_t)
        r = self.rel_embeddings(self.batch_r)
        r_inv = self.rel_inv_embeddings(self.batch_r)
        score = self._calc_avg(h, t, r, r_inv)
        regul = torch.mean(h ** 2) + torch.mean(t ** 2) + \
            torch.mean(r ** 2) + torch.mean(r_inv ** 2)
        return self.loss(score, regul)

    def predict(self):
        h = self.ent_embeddings(self.batch_h)
        t = self.ent_embeddings(self.batch_t)
        r = self.rel_embeddings(self.batch_r)
        score = self._calc_ingr(h, r, t)
        return score.cpu().data.numpy()
