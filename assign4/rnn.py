"""
Author: Wilhelm Ågren, wagren@kth.se
Last edited: 13/05-2021
"""

import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


class RNN:
    def __init__(self, save=True, sigma=0.1, m=100, eta=0.1, gamma=0.9, eps=np.finfo(float).eps, seq_length=25, verbose=False):
        self.FILEPATH = os.getcwd()
        self.BOOK_FILEPATH = os.path.join(self.FILEPATH, 'data\\goblet_book.txt')
        self.i2c_FILEPATH = os.path.join(self.FILEPATH, 'data\\vocab.p')
        self.DATA_FILEPATH = os.path.join(self.FILEPATH, 'data\\data.p')
        self.MODEL_FILEPATH = os.path.join(self.FILEPATH, 'model\\')
        self.vocab = set()
        self.book_data = []
        self.i2c = {}
        self.c2i = {}
        self.m = m
        self.K = 0
        self.N = 0  # After loading/reading data should be set to 1107545 for the given HP book.
        self.eta = eta
        self.sigma = sigma
        self.eps = eps
        self.gamma = gamma
        self.seq_length = seq_length
        self.b = np.zeros(1)
        self.c = np.zeros(1)
        self.U = np.zeros(1)
        self.W = np.zeros(1)
        self.V = np.zeros(1)
        self.grad_b = np.zeros(1)
        self.grad_c = np.zeros(1)
        self.grad_U = np.zeros(1)
        self.grad_W = np.zeros(1)
        self.grad_V = np.zeros(1)
        self.prev_m_b = 0
        self.prev_m_c = 0
        self.prev_m_W = 0
        self.prev_m_V = 0
        self.prev_m_U = 0
        self.h0 = np.zeros(1)
        self.hprev = np.zeros(1)
        self.x_chars = []
        self.y_chars = []
        self.loss_history = []
        self.save = save
        self.verbose = verbose
        print('-|' + '+'*100)

    def build_mapping(self, offset):
        for idx in range(self.seq_length):
            self.x_chars = [self.onehot_idx(self.c2i[self.book_data[idx]]) for idx in range(offset, offset+self.seq_length)]
            self.y_chars = [self.onehot_idx(self.c2i[self.book_data[idx + 1]]) for idx in range(offset, offset+self.seq_length)]

    def init_model(self):
        self.b = np.zeros((self.m, 1))
        self.c = np.zeros((self.K, 1))
        self.U = np.random.normal(0, self.sigma, size=(self.m, self.K))
        self.W = np.random.normal(0, self.sigma, size=(self.m, self.m))
        self.V = np.random.normal(0, self.sigma, size=(self.K, self.m))
        self.grad_b = np.zeros((self.m, 1))
        self.grad_c = np.zeros((self.K, 1))
        self.grad_U = np.zeros((self.m, self.K))
        self.grad_W = np.zeros((self.m, self.m))
        self.grad_V = np.zeros((self.K, self.m))
        self.h0 = np.zeros((self.m, 1))
        self.hprev = np.zeros((self.m, 1))
        self.x_chars = np.zeros((self.K, self.seq_length))
        self.y_chars = np.zeros((self.K, self.seq_length))

    def prepare_data(self):
        if self.verbose:
            print(f'<| preparing the data from filepath: {self.BOOK_FILEPATH}')
        if os.path.isfile(self.DATA_FILEPATH):
            self.__load__()
            assert len(self.vocab) == len(self.i2c.keys()), print(f'!!! ERROR :: vocab size not equal i2c size, {len(self.vocab)} != {len(self.i2c.keys())}')
            return
        with open(self.BOOK_FILEPATH, 'r', encoding='utf8') as fd:
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

    def forward(self, xt, yt, ht, b=None, c=None, W=None, V=None, U=None, grad=None) -> (list, list, list, list, float):
        """
        func forward/69
        @spec :: () => (list, list, list, list, float)
            Calculates the forward pass of the network/evaluation of the network, given the data sequence xt,
            target sequence yt, and initial state for hidden layer ht. Current activation function used is tanh,
            in order to avoid both under- and overflows in gradient computations. Testing with ReLU lead to instant
            overflows in gradient computations -> because the hidden states did not represent a code with values
            from [-1, 1], but was instead anything in [0, inf] because of the linear transformation in ReLU for anything
            larger than 0. The ReLU was suggested as another activation function in instructions but it did not work.
        """
        if b is None:
            b = self.b
        if c is None:
            c = self.c
        if W is None:
            W = self.W
        if V is None:
            V = self.V
        if U is None:
            U = self.U
        if grad is not None:
            for key in grad:
                if key == 'c':
                    c = grad[key]
                if key == 'b':
                    b = grad[key]
                if key == 'W':
                    W = grad[key]
                if key == 'V':
                    V = grad[key]
                if key == 'U':
                    U = grad[key]

        tau = len(xt)
        hl, pl, al, ol, ll = [], [], [], [], []
        hl.append(ht)
        for t in range(tau):
            al.append(W@hl[-1] + U@xt[t] + b)
            hl.append(np.tanh(al[-1]))
            ol.append(V@hl[-1] + c)
            pl.append(self.__softmax__(ol[-1]))
            ll.append(self.__loss__(pl[-1], yt[t]))
        hl.pop(0)
        return hl, pl, al, ol, sum(ll)

    def backprop(self, xt, yt, h) -> (int, np.array):
        """
        func backprop/4
        @spec :: (Class(RNN), np.array, np.array, np.array) => (integer, np.array)
            Back Propagation Through Time gradient computations for RNN.
            Takes the current sequence of data xt, corresponding targets ty, and last calculated state of h.
            Recursively calculates gradients of c, b, W, V, U. Clips gradients from [-5, 5] to avoid overflows.
        """
        tau = len(xt)
        ht, pt, at, ot, loss = self.forward(xt=xt, yt=yt, ht=h)

        assert len(ht) == tau
        assert len(pt) == tau
        assert len(at) == tau
        assert len(ot) == tau
        for (htt, ptt, att, ott) in zip(ht, pt, at, ot):
            assert htt.shape == (self.m, 1)
            assert ptt.shape == (self.K, 1)
            assert att.shape == (self.m, 1)
            assert ott.shape == (self.K, 1)

        grad_o_l, grad_h_l, grad_a_l = [np.zeros(1) for _ in range(tau)], \
                                       [np.zeros(1) for _ in range(tau)], \
                                       [np.zeros(1) for _ in range(tau)]

        grad_c = np.zeros((self.K, 1))
        grad_b = np.zeros((self.m, 1))
        grad_V = np.zeros((self.K, self.m))
        grad_W = np.zeros((self.m, self.m))
        grad_U = np.zeros((self.m, self.K))

        for t in reversed(range(tau)):
            grad_o_l[t] = -(yt[t] - pt[t]).T
            grad_c += grad_o_l[t].T
            grad_V += grad_o_l[t].T@ht[t].T

            if t == tau - 1:
                grad_h_l[t] = grad_o_l[t]@self.V
            else:
                grad_h_l[t] = grad_o_l[t]@self.V + grad_a_l[t + 1]@self.W
            grad_a_l[t] = grad_h_l[t]@np.diag(1 - (ht[t][:, 0]**2))

            grad_W += grad_a_l[t].T@ht[t - 1].T
            grad_U += grad_a_l[t].T@xt[t].T
            grad_b += grad_a_l[t].T

        grad_c = np.clip(a=grad_c, a_min=-5, a_max=5)
        grad_b = np.clip(a=grad_b, a_min=-5, a_max=5)
        grad_V = np.clip(a=grad_V, a_min=-5, a_max=5)
        grad_W = np.clip(a=grad_W, a_min=-5, a_max=5)
        grad_U = np.clip(a=grad_U, a_min=-5, a_max=5)

        self.grad_c = grad_c
        self.grad_b = grad_b
        self.grad_V = grad_V
        self.grad_W = grad_W
        self.grad_U = grad_U

        return loss, ht[-1]

    def fit(self, n_epochs=2, e=0) -> None:
        """
        func fit/3
        @spec :: (Class(RNN), integer, integer) => None
            Performs Stochastic Gradient Descent (SGD) according to the AdaGrad method.
            Iterates through the book-data list in steps of seq_length. For each
            seq_length, iterations are incremented. Whenever the entire book-data list
            is iterated, reset 'e' and increment num epochs.
                n_epochs specifies the maximum number of total epochs to train on, and
                is somewhat optional. Simply train for longer if you want.
        """
        print('<| adagrad initialized:')
        hprev, smooth_loss, epochs, iter = self.h0, None, 0, 0
        while epochs < n_epochs:
            if e >= len(self.book_data) - self.seq_length - 1:
                e = 0
                epochs += 1
                hprev = self.h0
            self.build_mapping(offset=e)

            loss, hprev = self.backprop(xt=self.x_chars, yt=self.y_chars, h=hprev)

            self.prev_m_c += self.grad_c**2
            self.c += -(self.grad_c * self.eta/(np.sqrt(self.prev_m_c + self.eps)))

            self.prev_m_b += self.grad_b ** 2
            self.b += -(self.grad_b * self.eta / (np.sqrt(self.prev_m_b + self.eps)))

            self.prev_m_W += self.grad_W ** 2
            self.W += -(self.grad_W * self.eta / (np.sqrt(self.prev_m_W + self.eps)))

            self.prev_m_V += self.grad_V ** 2
            self.V += -(self.grad_V * self.eta / (np.sqrt(self.prev_m_V + self.eps)))

            self.prev_m_U += self.grad_U ** 2
            self.U += -(self.grad_U * self.eta / (np.sqrt(self.prev_m_U + self.eps)))

            smooth_loss = loss if smooth_loss is None else 0.999*smooth_loss + 0.001*loss

            if iter % 100 == 0:
                print(f'\n\t\t iter = {iter},\tsmooth loss = {smooth_loss}')
            # if iter % 10000 == 0:
            #    self.synthesize(self.book_data[e], hl=hprev, n=200)
            #    time.sleep(1)
            e += self.seq_length
            iter += 1
            self.loss_history.append(smooth_loss)
            self.hprev = hprev

    def synthesize(self, x0, hl, n) -> None:
        """
        func synthesize/4
        @spec :: (Class(RNN), np.array, np.array, integer) => None
            Takes a given text sequence x0, optionally starting matrix h and number n of chars to generate.
            Computes the forward pass given x0 and h0 which yields softmaxed outputs,
            the probabilities are used to sample from the available characters to chose next character in sequence.
        """
        s = ''
        x0 = self.c2i[x0]
        for idx in range(n):
            hl, pt, _, _, _ = self.forward(xt=[self.onehot_idx(x0)], yt=[np.ones((self.K, 1))], ht=hl)
            rand_idx = np.random.choice(np.arange(0, self.K), p=pt[0].flatten('C'))
            hl = hl[-1]
            x0 = rand_idx
            s += self.i2c[x0]
        print(s)

    def numerical_grad(self, xt, yt, h=1e-4) -> None:
        """
        func numerical_grad/4
        @spec :: (Class(RNN), list, list, float) => None
            Computes the numerical gradients of each respective theta in the model.
            Does this the way God intended man to do it, using the definition of the derivative.
                =>    lim h->0 f(x+h) - f(x-h) / 2h
            Compares the gradients using self.__compare_gradients__/2 which is a relative difference of each element
            in the gradients. Otherwise can also compare using either mean or max absolute difference.
        """

        assert len(xt) == len(yt)

        weights = {'b': self.b, 'c': self.c, 'U': self.U, 'W': self.W, 'V': self.V}
        gradients = {'b': np.zeros(self.grad_b.shape), 'c': np.zeros(self.grad_c.shape),
                     'U': np.zeros(self.grad_U.shape), 'W': np.zeros(self.grad_W.shape), 'V': np.zeros(self.grad_V.shape)}
        analytical_gradients = {'b': self.grad_b, 'c': self.grad_c, 'W': self.grad_W, 'V': self.grad_V, 'U': self.grad_U}

        assert len(weights.keys()) == len(gradients.keys())
        for w_k, g_k in zip(weights.keys(), gradients.keys()):
            assert weights[w_k].shape == gradients[g_k].shape

        print('<| calculating numerical gradients for theta={b, c, U, W, V}')
        for weight in weights:
            grad = gradients[weight]
            for idx in range(grad.shape[0]):
                for jdx in range(grad.shape[1]):
                    grad_try = np.copy(grad)
                    grad_try[idx, jdx] += h
                    _, _, _, _, l1 = self.forward(xt=xt, yt=yt, ht=self.h0, grad={weight: grad_try})
                    grad_try = np.copy(grad)
                    grad_try[idx, jdx] -= h
                    _, _, _, _, l2 = self.forward(xt=xt, yt=yt, ht=self.h0, grad={weight: grad_try})
                    grad[idx, jdx] = (l1 - l2) / (2*h)

        for key in weights:
            analytical, numerical = analytical_gradients[key], gradients[key]
            # print(f'\t analytical gradient for  {key}  : {np.abs(analytical)}')
            # print(self.__compare_gradients__(analytical, numerical))
            print(f'\tmean err in gradient for  {key}  : {np.mean(np.abs(analytical - numerical))}\n')

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

    def __load_model__(self):
        """
        Loads the already trained model weights from file(s) using pickle for reading.
        Store the loaded weights as class attributes for Class(RNN).
        """
        if self.verbose:
            print(f'\t\tloading trained model weights from folder: {self.MODEL_FILEPATH}')
        self.b = pickle.load(open(os.path.join(self.MODEL_FILEPATH, 'b.p'), 'rb'))
        self.c = pickle.load(open(os.path.join(self.MODEL_FILEPATH, 'C.p'), 'rb'))
        self.W = pickle.load(open(os.path.join(self.MODEL_FILEPATH, 'W.p'), 'rb'))
        self.V = pickle.load(open(os.path.join(self.MODEL_FILEPATH, 'V.p'), 'rb'))
        self.U = pickle.load(open(os.path.join(self.MODEL_FILEPATH, 'U.p'), 'rb'))
        self.hprev = pickle.load(open(os.path.join(self.MODEL_FILEPATH, 'hprev.p'), 'rb'))

    def __save_model__(self):
        """
        Writes the trained model weights to file(s) using pickle for dumping.
        """
        if self.verbose:
            print(f'\t\twriting the model weights to folder: {self.MODEL_FILEPATH}')
        pickle.dump(self.b, open(os.path.join(self.MODEL_FILEPATH, 'b.p'), 'wb'))
        pickle.dump(self.c, open(os.path.join(self.MODEL_FILEPATH, 'c.p'), 'wb'))
        pickle.dump(self.W, open(os.path.join(self.MODEL_FILEPATH, 'W.p'), 'wb'))
        pickle.dump(self.V, open(os.path.join(self.MODEL_FILEPATH, 'V.p'), 'wb'))
        pickle.dump(self.U, open(os.path.join(self.MODEL_FILEPATH, 'U.p'), 'wb'))
        pickle.dump(self.hprev, open(os.path.join(self.MODEL_FILEPATH, 'hprev.p'), 'wb'))

    def __repr__(self) -> str:
        return 'Class(RNN)'

    def __del__(self) -> None:
        print(f'<| deleting instance of {self.__repr__()}')
        print('-|' + '+' * 100)

    def __plot_loss__(self) -> None:
        plt.plot([x for x in range(len(self.loss_history))], self.loss_history, linewidth=1, color='maroon')
        plt.xlabel('iterations')
        plt.ylabel('cross entropy loss')
        plt.title('RNN loss history')
        plt.show()

    @staticmethod
    def __loss__(p, y) -> float:
        """
        func __loss__/2
        @spec :: (np.array, np.array) => np.array
            Computes the cross-entropy loss between the given output probabilities p
            and the onehot encoded target labels y. Returns the loss as a np.array.
            If you want an integer, extract it from the array by accessing __loss__()[0, 0].
        """
        return -np.log(np.dot(y.T, p))[0, 0]

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
    def __compare_gradients__(g_a, g_n, eps=1e-6) -> np.array:
        """
        func __compare_gradients__/3
        @spec :: (np.array, np.array, float) => None
            Compare the given gradients. Analytical gradient g_a vs numerical gradient g_n.
            The difference between them is the absolute difference divided by the sum of elements.
        """
        assert (g_a.shape == g_n.shape), print('<| shape of g_a: {}\n<| shape of g_n: {}'.format(g_a.shape, g_n.shape))
        err = np.zeros(g_a.shape)
        for idx in range(err.shape[0]):
            for jdx in range(err.shape[1]):
                err[idx, jdx] = np.abs(g_a[idx, jdx] - g_n[idx, jdx]) / max(eps, np.abs(g_a[idx, jdx]) + np.abs(g_n[idx, jdx]))

        return err

    def __fit__(self, max_epoch=2, load=False):
        self.prepare_data()
        self.init_model()
        if load:
            self.__load_model__()
        self.fit(n_epochs=max_epoch)
        self.__plot_loss__()
        self.__save_model__()

    def __synthesize_from_trained__(self, n):
        self.prepare_data()
        self.init_model()
        self.__load_model__()
        self.build_mapping(offset=0)
        self.synthesize(x0=self.book_data[5], hl=self.hprev, n=n)


def main():
    rnn = RNN(eta=0.1, sigma=0.01, save=False, verbose=True, m=100, seq_length=25)
    # rnn.__fit__(load=True)
    rnn.__synthesize_from_trained__(1000)


if __name__ == '__main__':
    main()
