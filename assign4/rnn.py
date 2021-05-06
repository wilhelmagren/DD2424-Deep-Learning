"""
Author: Wilhelm Ågren, wagren@kth.se
Last edited: 05/05-2021
"""

import os
import pickle
import numpy as np
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
        self.h0 = np.zeros(1)
        self.x_chars = np.zeros((1, 1))
        self.y_chars = np.zeros((1, 1))
        self.save = save
        self.verbose = verbose
        print('-|' + '+'*100)

    def build_mapping(self):
        if self.verbose:
            print('<| building the onehot-encoded mappings x -> y')
        for idx in range(self.seq_length):
            self.x_chars[:, idx] = np.transpose(self.onehot_idx(self.c2i[self.book_data[idx]]).flatten())
            self.y_chars[:, idx] = self.onehot_idx(self.c2i[self.book_data[idx + 1]]).flatten()
        if self.verbose:
            print(f'\t\tx_chars shape: {self.x_chars.shape}')
            print(f'\t\ty_chars shape: {self.y_chars.shape}')

    def init_model(self):
        self.b = np.zeros((self.m, ))
        self.c = np.zeros((self.K, ))
        self.U = np.random.rand(self.m, self.K)*self.sigma
        self.W = np.random.rand(self.m, self.m)*self.sigma
        self.V = np.random.rand(self.K, self.m)*self.sigma
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

    def forward_pass(self, xt=None, yt=None, ht=None):
        if ht is None:
            ht = self.h0
        if xt is None:
            xt = self.x_chars
        if yt is None:
            yt = self.y_chars
        tao = xt.shape[1]
        h_l, p_l, a_l, o_l, loss_l = np.zeros((self.m, tao + 1)), np.zeros((self.K, tao)), \
                                     np.zeros((self.m, tao)), np.zeros((self.K, tao)), []
        h_l[:, 0] = ht
        for t in range(tao):
            # (m x 1) = (m x m)*(m x 1) + (m x k)*(d x 1) + (m x 1)   ::   Hidden state at time t before non-linearity
            a_l[:, t] = self.W@h_l[:, t] + self.U@xt[:, t] + self.b
            # hidden state at time t of size   (m x 1)
            h_l[:, t] = np.tanh(a_l[:, t])
            # output vector of unnormalized log probabilities for each class at time t of size   (K x 1)
            o_l[:, t] = self.V@h_l[:, t] + self.c
            # output probability vector at time t of size   (K x 1)
            p_l[:, t] = self.softmax(o_l[:, t])
            loss_l.append(self.loss(p_l[:, t], yt[:, t]))
        return h_l, p_l, a_l, o_l

    def adagrad(self, xt, yt):
        # grad_ot = -(y-p)^T
        # grad_pt = -(y^T)/(np.dot(y^t, pt))
        h_l, p_l, a_l, o_l = self.forward_pass(xt=xt, yt=yt)
        print(np.array(yt).shape)

    def synthesize(self, x0, n=None):
        if self.verbose:
            print(f'<| synthesizing a char sequence of length {n} from initial char {x0}')
        if n is None:
            n = self.seq_length
        if type(x0) is str:
            x0 = self.c2i[x0]
        s = '\t\t'
        for idx in range(n):
            _, pt, _, _ = self.forward_pass(xt=self.onehot_idx(x0))
            rand_idx = np.random.choice(np.arange(0, self.K), p=pt.flatten())
            x0 = rand_idx
            s += self.i2c[x0]
        print(s)

    def onehot_idx(self, idx):
        x = np.zeros((self.K, 1))
        x[idx, 0] = 1
        return x

    @staticmethod
    def loss(p, y):
        return -np.log(np.dot(y.T, p))

    @staticmethod
    def softmax(x):
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
    rnn = RNN(save=False, verbose=True, seq_length=5)
    rnn.prepare_data()
    rnn.init_model()
    rnn.build_mapping()
    # h, p, a, o = rnn.forward_pass(xt=rnn.x_chars, yt=rnn.y_chars)
    rnn.synthesize('q', n=10)
    # rnn.adagrad(xt=rnn.x_chars, yt=rnn.y_chars)


if __name__ == '__main__':
    main()
