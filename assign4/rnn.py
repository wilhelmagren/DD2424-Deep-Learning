"""
Author: Wilhelm Ågren, wagren@kth.se
Last edited: 10/05-2021
"""

import os
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class RNN:
    def __init__(self, save=True, sigma=0.01, m=100, eta=0.1, seq_length=25, verbose=False):
        self.FILEPATH = os.getcwd()
        self.BOOK_FILEPATH = os.path.join(self.FILEPATH, 'data\\goblet_book.txt')
        self.i2c_FILEPATH = os.path.join(self.FILEPATH, 'data\\vocab.p')
        self.DATA_FILEPATH = os.path.join(self.FILEPATH, 'data\\data.p')
        self.vocab = set()
        self.book_data = []
        self.i2c = {}
        self.c2i = {}
        self.m = m
        self.K = 0
        self.N = 0  # After loading/reading data should be set to 1107545 for the given HP book.
        self.eta = eta
        self.sigma = sigma
        self.seq_length = seq_length
        self.b = np.zeros(1)
        self.c = np.zeros(1)
        self.U = np.zeros(1)
        self.W = np.zeros(1)
        self.V = np.zeros(1)
        self.grad_U = np.zeros(1)
        self.grad_W = np.zeros(1)
        self.grad_V = np.zeros(1)
        self.h0 = np.zeros(1)
        self.x_chars = np.zeros((1, 1))
        self.y_chars = np.zeros((1, 1))
        self.save = save
        self.verbose = verbose
        print('-|' + '+'*100)

    def build_mapping(self, offset=0):
        for idx in range(self.seq_length):
            self.x_chars[:, idx] = np.transpose(self.onehot_idx(self.c2i[self.book_data[idx + offset]]).flatten())
            self.y_chars[:, idx] = self.onehot_idx(self.c2i[self.book_data[idx + 1 + offset]]).flatten()

    def init_model(self):
        self.b = np.zeros((self.m, ))
        self.c = np.zeros((self.K, ))
        self.U = np.random.rand(self.m, self.K)*self.sigma
        self.W = np.random.rand(self.m, self.m)*self.sigma
        self.V = np.random.rand(self.K, self.m)*self.sigma
        self.grad_U = np.zeros((self.m, self.m))
        self.grad_W = np.zeros((self.m, self.m))
        self.grad_V = np.zeros((self.K, self.m))
        self.h0 = np.zeros((self.m, ))
        self.x_chars = np.zeros((self.K, self.seq_length))
        self.y_chars = np.zeros((self.K, self.seq_length))

    def prepare_data(self):
        if self.verbose:
            print(f'<| preparing the data from filepath: {self.BOOK_FILEPATH}')
        if os.path.isfile(self.DATA_FILEPATH):
            self.__load()
            assert len(self.vocab) == len(self.i2c.keys()), print(f'!!! ERROR :: vocab size not equal i2c size, {len(self.vocab)} != {len(self.i2c.keys())}')
            return
        with open(self.BOOK_FILEPATH, 'r') as fd:
            for char in fd.read():
                self.book_data.append(char)
                if char not in self.vocab:
                    self.vocab.add(char)
                    self.i2c[len(self.vocab) - 1] = char
                    self.c2i[char] = len(self.vocab) - 1

        self.K = len(self.vocab)
        self.N = len(self.book_data)
        if self.save:
            self.__write()

        if self.verbose:
            print(f'<| prepared data features below:\n'
                  f'\t\tsize of vocab:\t{len(self.vocab)}\n'
                  f'\t\tsize of i2c:\t{len(self.i2c.keys())}\n'
                  f'\t\tsize of c2i:\t{len(self.c2i.keys())}\n'
                  f'\t\tsize of data:\t{len(self.book_data)}')

    def forward_pass(self, xt=None, yt=None, ht=None, W=None, V=None, U=None):
        if ht is None:
            ht = self.h0
        if xt is None:
            xt = self.x_chars
        if yt is None:
            yt = self.y_chars
        if W is None:
            W = self.W
        if V is None:
            V = self.V
        if U is None:
            U = self.U

        tau = xt.shape[1]
        h_l, p_l, a_l, o_l, loss_l = np.zeros((self.m, tau + 1)), np.zeros((self.K, tau)), \
                                     np.zeros((self.m, tau)), np.zeros((self.K, tau)), np.zeros((1, tau))
        h_l[:, 0] = ht
        for t in range(tau):
            # (m x 1) = (m x m)*(m x 1) + (m x k)*(d x 1) + (m x 1)   ::   Hidden state at time t before non-linearity
            a_l[:, t] = W@h_l[:, t] + U@xt[:, t] + self.b
            # hidden state at time t of size   (m x 1)
            h_l[:, t] = np.tanh(a_l[:, t])
            # output vector of unnormalized log probabilities for each class at time t of size   (K x 1)
            o_l[:, t] = V@h_l[:, t] + self.c
            # output probability vector at time t of size   (K x 1)
            p_l[:, t] = self.__softmax(o_l[:, t])
            loss_l[:, t] = self.__loss(p_l[:, t], yt[:, t])
        return h_l, p_l, a_l, o_l, loss_l

    def backprop(self, xt, yt, h):
        """
        Gradient computations for RNN, back propagation through time BPTT.
        """
        tau = xt.shape[1]
        ht, pt, at, ot, loss = self.forward_pass(xt=xt, yt=yt, ht=h)

        assert ht.shape == (self.m, tau + 1)
        assert pt.shape == (self.K, tau)
        assert at.shape == (self.m, tau)
        assert ot.shape == (self.K, tau)

        grad_o_l = np.zeros((tau, self.K))
        grad_p_l = np.zeros((tau, self.K))
        for t in range(tau):
            grad_p_l[t, :] = -yt[:, t].T/(yt[:, t].T@pt[:, t])
            grad_o_l[t, :] = -(yt[:, t] - pt[:, t]).T

        grad_V_l = np.zeros((self.K, self.m))
        for t in range(tau):
            grad_V_l += np.asmatrix(grad_o_l[t, :]).T@np.asmatrix(ht[:, t + 1])

        grad_h_l = np.zeros((tau, self.m))
        grad_a_l = np.zeros((tau, self.m))
        grad_h_l[tau - 1, :] = grad_o_l[tau - 1, :]@self.V
        grad_a_l[tau - 1, :] = grad_h_l[tau - 1, :]@np.diag(1 - (np.tanh(at[:, tau - 1])**2))
        for t in range(tau - 2, -1, -1):
            grad_h_l[t, :] = grad_o_l[t, :]@self.V + grad_a_l[t + 1, :]@self.W
            grad_a_l[t, :] = grad_h_l[t, :]@np.diag(1 - (np.tanh(at[:, t])**2))

        grad_W_l = np.zeros((self.m, self.m))
        grad_U_l = np.zeros((self.m, self.K))
        for t in range(tau):
            grad_W_l += np.asmatrix(grad_a_l[t, :]).T@np.asmatrix(ht[:, t])
            grad_U_l += np.asmatrix(grad_a_l[t, :]).T@np.asmatrix(xt[:, t])

        self.grad_V = grad_V_l
        self.grad_W = grad_W_l
        self.grad_U = grad_U_l

        return np.sum(loss), ht[:, tau]

    def adagrad(self, batch_n=100, e=0):
        """
        SGD training loop
        """
        print('<| adagrad initialized:')
        self.synthesize(self.book_data[0], n=200)
        hprev, smooth_loss = self.h0, None
        for iter in range(batch_n*len(self.book_data)):
            if e > len(self.book_data) - self.seq_length - 1:
                e = 0
                hprev = self.h0
            self.build_mapping(offset=e)
            loss, hprev = self.backprop(xt=self.x_chars, yt=self.y_chars, h=hprev)
            self.W += -self.eta*self.grad_W
            self.V += -self.eta*self.grad_V
            self.U += -self.eta*self.grad_U
            e += self.seq_length
            if smooth_loss is None:
                smooth_loss = loss
            smooth_loss = 0.999*smooth_loss + 0.001*loss
            if iter % 100 == 0:
                print(f'\n\t\t iter = {iter},\tsmooth loss = {smooth_loss}')
            if iter % 500 == 0:
                self.synthesize(self.book_data[e], h0=hprev, n=200)

    def synthesize(self, x0, h0=None, n=None):
        if n is None:
            n = self.seq_length
        if h0 is None:
            h0 = self.h0
        s = '\t\t'
        x0 = self.c2i[x0]
        for idx in range(n):
            _, pt, _, _, _ = self.forward_pass(xt=self.onehot_idx(x0), ht=h0)
            rand_idx = np.random.choice(np.arange(0, self.K), p=pt.flatten('C'))
            x0 = rand_idx
            s += self.i2c[x0]
        print(s)

    def onehot_idx(self, idx):
        x = np.zeros((self.K, 1))
        x[idx, 0] = 1
        return x

    def numerical_grad(self, x, y, h=1e-4):
        _, _, _, _, c = self.forward_pass(xt=x, yt=y)

        grad_W = np.zeros((self.m, self.m))
        grad_V = np.zeros((self.m, self.m))
        grad_U = np.zeros((self.m, self.m))
        try_W = self.W + h
        try_V = self.V + h
        try_U = self.U + h
        _, _, _, _, cW = self.forward_pass(xt=x, yt=y, W=try_W)
        _, _, _, _, cV = self.forward_pass(xt=x, yt=y, V=try_V)
        _, _, _, _, cU = self.forward_pass(xt=x, yt=y, U=try_U)
        for (costW, costV, costU) in zip(cW[0], cV[0], cU[0]):
            grad_W += (costW - c) / (2*h)
            grad_V += (costV - c) / (2 * h)
            grad_U += (costU - c) / (2 * h)

    @staticmethod
    def __loss(p, y):
        return -np.log(np.dot(y.T, p))

    @staticmethod
    def __softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def __load(self):
        if self.verbose:
            print(f'\t\tvocab already exists, loading it and building idx dictionaries ...')
        self.i2c = pickle.load(open(self.i2c_FILEPATH, 'rb'))
        for idx in self.i2c:
            self.c2i[self.i2c[idx]] = idx
            self.vocab.add(self.i2c[idx])
        self.K = len(self.vocab)
        self.book_data = pickle.load(open(self.DATA_FILEPATH, 'rb'))
        self.N = len(self.book_data)
        if self.verbose:
            print(f'<| loaded data features below:\n'
                  f'\t\tsize of vocab:\t{len(self.vocab)}\n'
                  f'\t\tsize of i2c:\t{len(self.i2c.keys())}\n'
                  f'\t\tsize of c2i:\t{len(self.c2i.keys())}\n'
                  f'\t\tsize of data:\t{len(self.book_data)}')

    def __write(self):
        if self.verbose:
            print(f'\t\twriting the i2c to filepath: {self.i2c_FILEPATH}')
            print(f'\t\twriting the data to filepath: {self.DATA_FILEPATH}')
        pickle.dump(self.i2c, open(self.i2c_FILEPATH, 'wb'))
        pickle.dump(self.book_data, open(self.DATA_FILEPATH, 'wb'))

    def __repr__(self):
        return 'Class(RNN)'

    def __del__(self):
        print(f'<| deleting instance of {self.__repr__()}')
        print('-|' + '+' * 100)


def main():
    np.random.seed(2)
    rnn = RNN(save=False, verbose=True, m=100, seq_length=25)
    rnn.prepare_data()
    rnn.init_model()
    rnn.build_mapping()
    rnn.adagrad()


if __name__ == '__main__':
    main()
