"""
Author: Wilhelm Ågren, wagren@kth.se
Last edited: 05/05-2021
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


class RNN:
    def __init__(self, sigma=0.01, m=100, eta=0.1, seq_length=25, verbose=False):
        self.FILEPATH = os.getcwd()
        self.DATA_FILEPATH = os.path.join(self.FILEPATH, 'data\\goblet_book.txt')
        self.VOCAB_FILEPATH = os.path.join(self.FILEPATH, 'data\\vocab.p')
        self.vocab = set()
        self.book_data = []
        self.i2c = {}
        self.c2i = {}
        self.m = m
        self.K = 0
        self.eta = eta
        self.sigma = sigma
        self.seq_length = seq_length
        self.b = np.zeros(1)
        self.c = np.zeros(1)
        self.U = np.zeros(1)
        self.W = np.zeros(1)
        self.V = np.zeros(1)
        self.h0 = np.zeros(1)
        self.verbose = verbose

    def init_model(self):
        self.b = np.zeros((self.m, 1))
        self.c = np.zeros((self.K, 1))
        self.U = np.random.rand(self.m, self.K)*self.sigma
        self.W = np.random.rand(self.m, self.m)*self.sigma
        self.V = np.random.rand(self.K, self.m)*self.sigma
        self.h0 = np.zeros((self.m, 1))

    def prepare_data(self):
        if self.verbose:
            print(f'<| preparing the data from filepath: {self.DATA_FILEPATH}')
        if os.path.isfile(self.VOCAB_FILEPATH):
            self.__load_vocab()
            return
        with open(self.DATA_FILEPATH, 'r') as fd:
            for char in fd.read():
                if char not in self.vocab:
                    self.vocab.add(char)
                    self.i2c[len(self.vocab) - 1] = char
                    self.c2i[char] = len(self.vocab) - 1

        self.K = len(self.vocab)
        self.__write_vocab()

    def __load_vocab(self):
        if self.verbose:
            print(f'<| vocab already exists, loading it and building idx dictionaries')
        self.i2c = pickle.load(open(self.VOCAB_FILEPATH, 'rb'))
        for idx in self.i2c:
            self.c2i[self.i2c[idx]] = idx
            self.vocab.add(self.i2c[idx])
        self.K = len(self.vocab)

    def __write_vocab(self):
        if self.verbose:
            print(f'<| writing the vocab to filepath: {self.VOCAB_FILEPATH}')
        pickle.dump(self.i2c, open(self.VOCAB_FILEPATH, 'wb'))

    def loss(self, x, y):
        pass

    def forward_pass(self, xt, tao=2):
        h_l = [self.h0]
        p_l = []
        for t in range(1, tao):
            # (m x 1) = (m x m)*(m x 1) + (m x k)*(d x 1) + (m x 1)   ::   Hidden state at time t before non-linearity
            at = self.W@h_l[t-1] + self.U@xt + self.b
            # hidden state at time t of size   (m x 1)
            h_l.append(np.tanh(at))
            # output vector of unnormalized log probabilities for each class at time t of size   (K x 1)
            ot = self.V@h_l[t] + self.c
            # output probability vector at time t of size   (K x 1)
            p_l.append(self.softmax(ot))
        return h_l, p_l

    def synthesize(self, x0, n=None):
        if n is None:
            n = self.seq_length
        if type(x0) is str:
            x0 = self.c2i[x0]
        if self.verbose:
            print(f'<| synthesizing a char sequence of length {n} from initial char {x0}')
        s = ''
        for idx in range(n):
            _, p_l = self.forward_pass(self.onehot_idx(x0), tao=2)
            rand_idx = np.random.choice(np.arange(0, self.K), p=p_l[0].flatten())
            s += self.i2c[x0]
            x0 = rand_idx
        print(s)

    def onehot_idx(self, idx):
        x = np.zeros((self.K, 1))
        x[idx, 0] = 1
        return x

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)


def main():
    rnn = RNN(verbose=True)
    rnn.prepare_data()
    rnn.init_model()
    # rnn.forward_pass(tmp_x)
    rnn.synthesize('.', n=5)


if __name__ == '__main__':
    main()
