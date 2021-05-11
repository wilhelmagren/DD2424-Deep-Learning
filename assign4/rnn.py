"""
Author: Wilhelm Ågren, wagren@kth.se
Last edited: 11/05-2021
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
        self.grad_U = np.zeros((self.m, self.K))
        self.grad_W = np.zeros((self.m, self.m))
        self.grad_V = np.zeros((self.K, self.m))
        self.h0 = np.zeros((self.m, ))
        self.x_chars = np.zeros((self.K, self.seq_length))
        self.y_chars = np.zeros((self.K, self.seq_length))

    def prepare_data(self):
        if self.verbose:
            print(f'<| preparing the data from filepath: {self.BOOK_FILEPATH}')
        if os.path.isfile(self.DATA_FILEPATH):
            self.__load__()
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
            self.__write__()

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
            p_l[:, t] = self.__softmax__(o_l[:, t])
            loss_l[:, t] = self.__loss__(p_l[:, t], yt[:, t])
        return h_l, p_l, a_l, o_l, np.sum(loss_l)

    def backprop(self, xt, yt, h) -> (int, np.array):
        """
        func backprop/4
        @spec :: (Class(RNN), np.array, np.array, np.array) => (integer, np.array)
            Back Propagation Through Time gradient computations for RNN.

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

        return loss, ht[:, tau]

    def adagrad(self, n_epochs=100, e=0) -> None:
        """
        func adagrad/3
        @spec :: (Class(RNN), integer, integer) => None
            Performs Stochastic Gradient Descent (SGD) according to the AdaGrad method.
            Iterates through the book-data list in steps of seq_length. For each
            seq_length, iterations are incremented. Whenever the entire book-data list
            is iterated, reset 'e' and increment num epochs.
                n_epochs specifies the maximum number of total epochs to train on, and
                is somewhat optional. Simply train for longer if you want.
        """
        print('<| adagrad initialized:')
        hprev, smooth_loss, epochs, smooth_loss = self.h0, None, 0, 100
        while epochs < n_epochs:
            if e > len(self.book_data) - self.seq_length - 1:
                e = 0
                epochs += 1
                hprev = self.h0
            self.build_mapping(offset=e)
            loss, hprev = self.backprop(xt=self.x_chars, yt=self.y_chars, h=hprev)
            self.W += -self.eta*self.grad_W
            self.V += -self.eta*self.grad_V
            self.U += -self.eta*self.grad_U
            smooth_loss = 0.999*smooth_loss + 0.001*loss
            if e % 100 == 0:
                print(f'\n\t\t iter = {e},\tsmooth loss = {smooth_loss}')
            if e % 500 == 0:
                self.synthesize(self.book_data[e], h0=hprev, n=200)
            e += self.seq_length

    def synthesize(self, x0, h0=None, n=None) -> None:
        """
        func synthesize/4
        @spec :: (Class(RNN), np.array, np.array, integer) => None
            Takes a given text sequence x0, optionally starting matrix h and number n of chars to generate.
            Computes the forward pass given x0 and h0 which yields softmaxed outputs,
            the probabilities are used to sample from the available characters to chose next character in sequence.
        """
        if n is None:
            n = self.seq_length
        if h0 is None:
            h0 = self.h0
        s = ''
        x0 = self.c2i[x0]
        for idx in range(n):
            _, pt, _, _, _ = self.forward_pass(xt=self.onehot_idx(x0), ht=h0)
            rand_idx = np.random.choice(np.arange(0, self.K), p=pt.flatten('C'))
            x0 = rand_idx
            s += self.i2c[x0]
        print(s)

    def numerical_grad(self, x, y, h=1e-4) -> (np.array, np.array, np.array):
        """
        Don't look at this abomination...
        """

        assert x.shape == y.shape

        grad_U = np.zeros((self.m, self.K))
        grad_W = np.zeros((self.m, self.m))
        grad_V = np.zeros((self.K, self.m))
        grad_n_l = [grad_U, grad_W, grad_V]
        grad_a_l = [self.grad_U, self.grad_W, self.grad_V]

        print('<| calculating numerical gradients for theta={U, W, V}')
        print('\t\tInitiating grad_U numerical')
        for idx in tqdm(range(grad_U.shape[0])):
            for jdx in range(grad_U.shape[1]):
                grad_try_U = np.copy(grad_U)
                grad_try_U[idx, jdx] += h
                _, _, _, _, loss_plus = self.forward_pass(xt=x, yt=y, U=grad_try_U)

                grad_try_U = np.copy(grad_U)
                grad_try_U[idx, jdx] -= h
                _, _, _, _, loss_minus = self.forward_pass(xt=x, yt=y, U=grad_try_U)
                grad_U[idx, jdx] = (loss_plus - loss_minus) / (2*h)

        print('\n\t\tInitiating grad_W numerical')
        for idx in tqdm(range(grad_W.shape[0])):
            for jdx in range(grad_W.shape[1]):
                grad_try_W = np.copy(grad_W)
                grad_try_W[idx, jdx] += h
                _, _, _, _, loss_plus = self.forward_pass(xt=x, yt=y, W=grad_try_W)

                grad_try_W = np.copy(grad_W)
                grad_try_W[idx, jdx] -= h
                _, _, _, _, loss_minus = self.forward_pass(xt=x, yt=y, W=grad_try_W)
                grad_W[idx, jdx] = (loss_plus - loss_minus) / (2*h)

        print('\n\t\tInitiating grad_V numerical')
        for idx in tqdm(range(grad_V.shape[0])):
            for jdx in range(grad_V.shape[1]):
                grad_try_V = np.copy(grad_V)
                grad_try_V[idx, jdx] += h
                _, _, _, _, loss_plus = self.forward_pass(xt=x, yt=y, V=grad_try_V)

                grad_try_V = np.copy(grad_V)
                grad_try_V[idx, jdx] -= h
                _, _, _, _, loss_minus = self.forward_pass(xt=x, yt=y, V=grad_try_V)
                grad_V[idx, jdx] = (loss_plus - loss_minus) / (2*h)

        for (g_a, g_n) in zip(grad_a_l, grad_n_l):
            self.__compare_gradients__(g_a=g_a, g_n=g_n)

        return grad_V, grad_W, grad_U

    def onehot_idx(self, idx) -> np.array:
        """
        func onehot_idx/2
        @spec :: (Class(RNN), integer) => np.array
            Takes the given character index, respective to self.c2i, and instantiates
            a np.array of the onehot encoded character index. Simply returns it. Bye.
        """
        x = np.zeros((self.K, 1))
        x[idx, 0] = 1
        return x

    @staticmethod
    def __loss__(p, y) -> np.array:
        """
        func __loss__/2
        @spec :: (np.array, np.array) => np.array
            Computes the cross-entropy loss between the given output probabilities p
            and the onehot encoded target labels y. Returns the loss as a np.array.
            If you want an integer, extract it from the array by accessing __loss__()[0, 0].
        """
        return -np.log(np.dot(y.T, p))

    @staticmethod
    def __softmax__(x) -> np.array:
        """
        func __softmax__/1
        @spec :: (np.array) => np.array
            Computes the softmax of the given np.array x.
            This turns the values of x into probabilities such that they sum to 1.
        """
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    @staticmethod
    def __compare_gradients__(g_a, g_n, eps=1e-6):
        assert (g_a.shape == g_n.shape), print('<| shape of g_a: {}\n<| shape of g_n: {}'.format(g_a.shape, g_n.shape))
        for idx in range(g_a.shape[0]):
            s = '['
            for jdx in range(g_a.shape[1]):
                s += ' ' + str(np.abs(g_a[idx, jdx] - g_n[idx, jdx]) / max(eps, np.abs(g_a[idx, jdx]) + np.abs(g_n[idx, jdx])))
            print(s + ']')

    def __load__(self) -> None:
        """
        Load the current model, i.e. i2c dict and book-data list, from the saved files
        and store them as class attributes. Using pickle to load.
        """
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

    def __write__(self) -> None:
        """
        Write the current model, i.e. i2c dict and book-data list, to files using pickle for dumping.
        """
        if self.verbose:
            print(f'\t\twriting the i2c to filepath: {self.i2c_FILEPATH}')
            print(f'\t\twriting the data to filepath: {self.DATA_FILEPATH}')
        pickle.dump(self.i2c, open(self.i2c_FILEPATH, 'wb'))
        pickle.dump(self.book_data, open(self.DATA_FILEPATH, 'wb'))

    def __repr__(self) -> str:
        return 'Class(RNN)'

    def __del__(self) -> None:
        print(f'<| deleting instance of {self.__repr__()}')
        print('-|' + '+' * 100)


def main():
    np.random.seed(2)
    rnn = RNN(save=False, verbose=True, m=100, seq_length=25)
    rnn.prepare_data()
    rnn.init_model()
    rnn.build_mapping()
    rnn.backprop(xt=rnn.x_chars, yt=rnn.y_chars, h=rnn.h0)
    _ = rnn.numerical_grad(x=rnn.x_chars, y=rnn.y_chars)

    # rnn.adagrad()


if __name__ == '__main__':
    main()
