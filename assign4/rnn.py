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
        self.verbose = verbose
        print('+'*100)

    def init_model(self):
        self.b = np.zeros((self.m, 1))
        self.c = np.zeros((self.K, 1))
        self.U = np.random.rand(self.m, self.K)*self.sigma
        self.W = np.random.rand(self.m, self.m)*self.sigma
        self.V = np.random.rand(self.K, self.m)*self.sigma
        self.h0 = np.zeros((self.m, 1))

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
        self.__write()

        if self.verbose:
            print(f'<| prepared data features below:\n'
                  f'\t\tsize of vocab:\t{len(self.vocab)}\n'
                  f'\t\tsize of i2c:\t{len(self.i2c.keys())}\n'
                  f'\t\tsize of c2i:\t{len(self.c2i.keys())}\n'
                  f'\t\tsize of data:\t{len(self.book_data)}')

    def loss(self, x, y):
        pass

    def cross_entropy(self, pt, yt):
        pass

    def forward_pass(self, xt, ht=None):
        if ht is None:
            ht = self.h0
        # (m x 1) = (m x m)*(m x 1) + (m x k)*(d x 1) + (m x 1)   ::   Hidden state at time t before non-linearity
        at = self.W@ht + self.U@xt + self.b
        # hidden state at time t of size   (m x 1)
        ht = np.tanh(at)
        # output vector of unnormalized log probabilities for each class at time t of size   (K x 1)
        ot = self.V@ht + self.c
        # output probability vector at time t of size   (K x 1)
        pt = self.softmax(ot)
        return ht, pt

    def synthesize(self, x0, n=None):
        if self.verbose:
            print(f'<| synthesizing a char sequence of length {n} from initial char {x0}')
        if n is None:
            n = self.seq_length
        if type(x0) is str:
            x0 = self.c2i[x0]
        s = '\t\t'
        for idx in range(n):
            ht, pt = self.forward_pass(self.onehot_idx(x0))
            rand_idx = np.random.choice(np.arange(0, self.K), p=pt.flatten())
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
        print('+' * 100)


def main():
    rnn = RNN(verbose=True)
    rnn.prepare_data()
    rnn.init_model()
    # rnn.forward_pass(tmp_x)
    rnn.synthesize('k', n=5)


if __name__ == '__main__':
    main()
