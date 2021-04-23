"""
Author: Wilhelm Ågren, wagren@kth.se
Last edited: 19/04/2021
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# ======================================================================================================================
def parse_data(dset, verbose=False):
    """
    reformat the given dataset such that it returns the data, the one-hot encoded labels,
    and the labels.
    output:
            X := d x n size which contains the image pixel data.
                 n is number of images, d is dimensionality of each image
            Y := one hot encoded labels, K x n matrix
            y := vector length n containing label for each image.
    """
    y = np.array(dset[b'labels'])
    X = dset[b'data'].T
    K = len(np.unique(y))
    n = len(X[0])
    Y = np.zeros((K, n))
    for idx, num in enumerate(y):
        Y[num, idx] = 1

    if verbose:
        print('\tthe shape of X:', X.shape)
        print('\tthe shape of Y:', Y.shape)
        print('\tthe shape of y:', y.shape)

    return X, Y, y


def load_batch(filename):
    import pickle
    with open('Dataset/' + filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)
# ======================================================================================================================


# noinspection PyTypeChecker
class KNN:

    def __init__(self, X=None, Y=None, y=None,
                 X_eval=None, Y_eval=None, y_eval=None,
                 X_test=None, Y_test=None, y_test=None,
                 batch_size=0, n_epochs=1, eta=0.0, lamda=0.0,
                 num_layers=0, num_nodes=None, shuffle=False,
                 batch_norm=False, dimensionality=3072, verbose=False):
        if num_nodes is None:
            num_nodes = []
        self.X = X
        self.Y = Y
        self.y = y
        self.X_eval = X_eval
        self.Y_eval = Y_eval
        self.y_eval = y_eval
        self.X_test = X_test
        self.Y_test = Y_test
        self.y_test = y_test
        # We need a weight matrix and bias vector for each layer of the neural network
        self.W = [np.zeros((1, 1)) for _ in range(num_layers)]
        self.b = [np.zeros((1, 1)) for _ in range(num_layers)]
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.eta = eta
        self.lamda = lamda
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.hidden_layers = num_layers - 1
        self.num_samples = 0
        # Initialize the gamma and beta hyperparameters used for batch normalization
        self.gamma = [1 for _ in range(len(num_nodes))]
        self.beta = [0 for _ in range(len(num_nodes))]
        self.grad_w = None
        self.grad_b = None
        self.training_mean = 0
        self.training_std = 1
        self.dimensionality = dimensionality
        self.shuffle = shuffle
        self.batch_norm = batch_norm
        self.verbose = verbose

    @staticmethod
    def batch_normalize(S, mu, V, eps=1e-4):
        if len(S.shape) < 2:
            S = np.asmatrix(S).T
        if len(mu.shape) < 2:
            mu = np.asmatrix(mu).T
        if len(V.shape) < 2:
            V = np.asmatrix(V).T
        return (S - mu) / np.sqrt(np.diag(V + eps))

    def preprocess_data(self, x):
        print('<| Preprocessing data ...')
        normalized = (x - self.training_mean) / self.training_std
        return normalized

    def parse_full_data(self, val_split=5000):
        """
        Func parse_full_data/2
        @spec (self, int) :: ()
            Loads the batch data, parses it and stores it accordingly in the KNN class datastructure.
            Preprocesses the X data, i.e. the images such that they are normalized by the mean and std,
        """
        if self.verbose:
            print('<| Parsing all of the batches ...')

        dataX, dataY, datay = parse_data(load_batch('data_batch_1'))
        dataX2, dataY2, datay2 = parse_data(load_batch('data_batch_2'))
        dataX3, dataY3, datay3 = parse_data(load_batch('data_batch_3'))
        dataX4, dataY4, datay4 = parse_data(load_batch('data_batch_4'))
        dataX5, dataY5, datay5 = parse_data(load_batch('data_batch_5'))
        X, Y, y = np.concatenate((dataX, dataX2, dataX3, dataX4, dataX5[:, :val_split]), axis=1), \
                  np.concatenate((dataY, dataY2, dataY3, dataY4, dataY5[:, :val_split]), axis=1), \
                  np.concatenate((datay, datay2, datay3, datay4, datay5[:val_split]))
        eval_X, eval_Y, eval_y = dataX5[:, val_split:], dataY5[:, val_split:], datay5[val_split:]
        self.training_std = X.std(axis=1).reshape(X.shape[0], 1)
        self.training_mean = X.mean(axis=1).reshape(X.shape[0], 1)
        self.X = self.preprocess_data(X)[:self.dimensionality, :]
        self.Y = Y[:self.dimensionality, :]
        self.y = y
        self.X_eval = self.preprocess_data(eval_X)[:self.dimensionality, :]
        self.Y_eval = eval_Y[:self.dimensionality, :]
        self.y_eval = eval_y
        self.num_samples = X.shape[1]

        test_X, test_Y, test_y = parse_data(load_batch('test_batch'))
        self.X_test = self.preprocess_data(test_X)[:self.dimensionality, :]
        self.Y_test = test_Y[:self.dimensionality, :]
        self.y_test = test_y

        if self.verbose:
            print(f'\t\tThe shape of X: {self.X.shape}')
            print(f'\t\tThe shape of X_eval: {self.X_eval.shape}')
            print(f'\t\tThe shape of X_test: {self.X_test.shape}')

    def initialize_params(self, d=None):
        """

        """
        assert self.num_layers == len(self.num_nodes) + 1, print(f'<| ERROR: num_layers={self.num_layers} '
                                                                 f'and we got {len(self.num_nodes)} hidden nodes...')
        #  K = Y.shape[0], d = X.shape[0], m = hid_nodes
        if d is None:
            d = self.X.shape[0]
        # The first weight matrix will always have size (m, d). Where m = num_nodes[0]
        # We will always have a first weight matrix, since this class requires k-layer network where k >= 2
        self.W[0] = np.random.normal(0, 1 / np.sqrt(self.X.shape[0]), size=(self.num_nodes[0], d))
        for num_node in range(1, len(self.num_nodes)):
            self.W[num_node] = np.random.normal(0, 1 / np.sqrt(self.num_nodes[num_node]),
                                                size=(self.num_nodes[num_node], self.num_nodes[num_node - 1]))

        self.W[self.num_layers - 1] = np.random.normal(0, 1 / np.sqrt(self.num_nodes[-1]),
                                                       size=(self.Y.shape[0], self.num_nodes[-1]))

        for idx, num_node in enumerate(self.num_nodes):
            self.b[idx] = np.zeros(shape=(num_node, 1))
        self.b[-1] = np.zeros(shape=(self.Y.shape[0], 1))

        if self.verbose:
            print('<| Initializing the network parameters ...')
            for idx, w in enumerate(self.W):
                print(f'\t\tthe shape of W{idx + 1}: {w.shape}')
            for idx, b in enumerate(self.b):
                print(f'\t\tthe shape of b{idx + 1}: {b.shape}')
        return

    def forward_pass(self, X=None, W=None, b=None):
        """
        func forward_pass/2
        @spec (self, np.array()) :: np.array(), np.array()
            Performs the forward pass of the network model, which corresponds to evaluating the network.
            Simply propagate your initial data through the network and perform all calculations with the
            model parameters W and b. Keep track of your updated scores S and data points X.
        """

        # assign a temporary variable which holds the propagated input
        if X is None:
            X = self.X
        if W is None:
            W = self.W
        if b is None:
            b = self.b
        # Transform the data to a matrix
        if len(X.shape) < 2:
            X = np.asmatrix(X).T
        X_tmp = X
        S, H = [], []
        for i in range(self.hidden_layers):
            # This is the un-normalized score for layer l
            S.append(W[i] @ X_tmp + b[i])
            H.append(np.maximum(0, S[-1]))
            X_tmp = H[-1]
        # Apply the final linear transformation
        S.append(W[-1] @ H[-1] + b[-1])
        # Apply softmax operation to turn final scores into probabilities
        P = softmax(S[-1])

        return P, S, H

    def forward_pass_BN(self, X=None):
        """
        func forward_pass_BN/2
        @spec (self, np.array()) :: np.array(), np.array()
            Performs the forward pass of the network model, with batch normalization.
            TODO: actually implement batch normalization...
        """
        # assign a temporary variable which holds the propagated input
        if X is None:
            X = self.X
        # Transform the data to a matrix
        if len(X.shape) < 2:
            X = np.asmatrix(X).T
        # From the forward pass of the back-prop algorithm you should store
        # X_b^l = (x_1^l, x_2^l, ..., x_n^l), S_b^l = (s_1^l, s_2^l, ..., s_n^l) and normalized S_b^l
        S_l, S_norm_l, mu, std, H, P, X_tmp = [], [], [], [], [], [], X
        for i in range(self.hidden_layers):
            # This is the un-normalized score for layer l
            S_l.append(self.W[i] @ X_tmp + self.b[i])
            mu.append(np.mean(S_l[-1], axis=1))
            std.append(np.var(S_l[-1], axis=1))
            S_norm_l.append(self.batch_normalize(S_l[-1], mu[-1], std[-1]))
            H.append(np.maximum(0, np.multiply(self.gamma[i], S_norm_l[-1]) + self.beta[i]))
            X_tmp = H[-1]

        # Apply the final linear transformation
        S_l.append(self.W[-1]@H[-1] + self.b[-1])
        P = softmax(softmax(S_l[-1]))

        return P, S_l, S_norm_l, H, mu, std

    def backward_pass(self, X=None, Y=None):

        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y

        if len(X.shape) < 2:
            X = np.asmatrix(X).T
        if len(Y.shape) < 2:
            Y = np.asmatrix(Y).T

        P_batch, S_batch, X_l = self.forward_pass(X)
        grad_W_l, grad_b_l = [], []

        G_batch = -(Y - P_batch)
        for l in range(self.hidden_layers, 0, -1):
            grad_W_l.append((G_batch @ X_l[l - 1].T) / X_l[l - 1].shape[1] + 2*self.lamda*self.W[l])
            grad_b_l.append(np.reshape((1 / X.shape[1]) * G_batch @ np.ones(X.shape[1]), (self.b[l].shape[0], 1)))
            G_batch = self.W[l].T @ G_batch
            G_batch = np.multiply(G_batch, X_l[l - 1] > 0)

        grad_W_l.append(((G_batch@X.T) / X.shape[1]) + 2*self.lamda*self.W[0])
        grad_b_l.append(np.reshape((1 / X.shape[1]) * G_batch @ np.ones(X.shape[1]), (self.b[0].shape[0], 1)))

        # Does this even work?
        grad_W_l.reverse()
        grad_b_l.reverse()
        self.grad_w = grad_W_l
        self.grad_b = grad_b_l

        return grad_W_l, grad_b_l

    def compute_grads_num_slow(self, X, Y, h=1e-5):
        """
        Computes the gradients the way that God intended man to do it.
        This is simply the definition of the derivative.
        lim h->inf (f(x+h) - f(x))/h
        """
        grad_W_l, grad_b_l, grad_gamma_l, grad_beta_l = [], [], [], []
        for b, W in zip(self.b, self.W):
            grad_b_l.append(np.zeros(shape=b.shape))
            grad_W_l.append(np.zeros(shape=W.shape))

        # I'm not sure on how this should work...
        if self.batch_norm:
            grad_gamma_l.append(np.zeros(len(self.gamma)))
            grad_beta_l.append(np.zeros(len(self.beta)))

        # Iterate over all the bias vectors
        for j in range(len(self.b)):
            for i in range(self.b[j].shape[0]):
                b_try = self.b
                b_try[j][i] = b_try[j][i] - h
                c1, _ = self.compute_cost(X, Y, b=b_try)

                b_try = self.b
                b_try[j][i] = b_try[j][i] + h
                c2, _ = self.compute_cost(X, Y, b=b_try)

                grad_b_l[j][i] = (c2 - c1)/(2*h)

        # Iterate over all the weight matrices
        for j in range(len(self.W)):
            for i in range(self.W[j].shape[0]):
                W_try = self.W
                W_try[j][i] = W_try[j][i] - h
                c1, _ = self.compute_cost(X, Y, W=W_try)

                W_try = self.W
                W_try[j][i] = W_try[j][i] + h
                c2, _ = self.compute_cost(X, Y, W=W_try)

                grad_W_l[j][i] = (c2 - c1)/(2*h)

        if self.batch_norm:
            pass

        return grad_W_l, grad_b_l, grad_gamma_l, grad_beta_l

    def compute_cost(self, X, Y, W=None, b=None):
        """
        func compute_cost/3
        @spec (self, np.array(), np.array()) :: float, float
            Computes the cost and loss given the network parameters W and the data X with corresponding labels Y.
            J(B, Lambda, THETA) = 1/n SUM{from i=1 => n} l_cross(x_i, y_i, THETA) + lambda*SUM{from i=1 => k} ||W_i||^2
            Features L2 regularization, and the cost is the sum of the loss and the regularization term. The loss
            is ONLY the sum of the l_cross term.
        """
        assert len(Y.shape) > 1, print('<| ERROR: labels are not one-hot encoded ...')
        # Transform the data to a matrix
        if len(X.shape) < 2:
            X = np.asmatrix(X).T

        if W is None:
            W = self.W
        if b is None:
            b = self.b

        loss_sum, reg_sum = 0, 0
        for w in W:
            reg_sum += self.lamda * np.sum(w ** 2)

        P, _, _ = self.forward_pass(X, W=W, b=b)
        for col in range(X.shape[1]):
            loss_sum += -np.log(np.dot(Y[:, col].T, P[:, col]))
        loss = loss_sum / X.shape[1]
        cost = loss + reg_sum
        return cost, loss

    def compute_acc(self, X, y, label='training', verbose=False):
        """
        func compute_acc/5
        @spec (self, np.array(), np.array(), str) :: float
            Computes the accuracy given the found model parameters W, on the data X and corresponding labels y.
            The y labels are not one hot encoded now. Calculates the forward pass in order to evaluate the network.
        """
        # Transform the data to a matrix in the case of X simply being one data sample
        if len(X.shape) < 2:
            X = np.asmatrix(X).T
        correct = 0
        P, _, _ = self.forward_pass(X)
        for k in range(y.shape[0]):
            pred = np.argmax(P[:, k])
            if pred == y[k]:
                correct += 1

        if verbose:
            print(f'<| Computed {label} accuracy on {self.num_layers}-NN is: {(round(correct / y.shape[0], 4))*100}%')

        return correct / y.shape[0]

    def fit(self, X, Y, y):
        """
        func fit/4
        @spec fit(self, np.array(), np.array(), np.array()) :: list, list, list, list, list, list, list
            Trains the model parameters self.W and self.b to the data X, Y and y. Utilizes mini-batch gradient descent
            and features shuffling of the training data prior to each epoch. Cyclic learning rate is used and the
            model is only trained for as many cycles that are specified. If all learning rate cycles have been performed
            before all epochs have been iterated, training will be terminated.
        """
        cost_training, loss_training, cost_eval, loss_eval, acc_t, eta_l, acc_e = [], [], [], [], [], [], []
        learning_rate, eta_min, eta_max, l, t = 1e-3, 1e-5, 1e-1, 0, 1
        num_cycle, n_s = 2, 5 * 45000 / self.batch_size
        breakk = False
        assert (X.shape[1] % self.batch_size == 0), print('<| CAN NOT SPLIT DATA ACCORDINGLY')
        # --------------------------------------------------------------------------------------------------------------
        # Training loss
        t_c, t_l = self.compute_cost(X, Y)
        cost_training.append(t_c)
        loss_training.append(t_l)
        # Eval loss
        e_c, e_l = self.compute_cost(self.X_eval, self.Y_eval)
        cost_eval.append(e_c)
        loss_eval.append(e_l)
        # Accuracy
        acc_t.append(self.compute_acc(X, y, label='training'))
        acc_e.append(self.compute_acc(self.X_eval, self.y_eval, label='validation'))
        # --------------------------------------------------------------------------------------------------------------
        for _ in tqdm(range(self.n_epochs)):
            if breakk:
                break
            # Shuffle the data
            if self.shuffle:
                indices = np.arange(X.shape[1])
                np.random.shuffle(indices)
                X = X[:, indices]
                Y = Y[:, indices]
                y = y[indices]
            for j in range(int(X.shape[1]/self.batch_size)):
                if 2 * l * n_s <= t <= (2 * l + 1) * n_s:
                    # Learning increasing
                    learning_rate = eta_min + (eta_max - eta_min) * (t - 2 * l * n_s) / n_s
                elif (2 * l + 1) * n_s <= t <= 2 * (l + 1) * n_s:
                    # Learning rate decreasing
                    learning_rate = eta_max - (t - (2 * l + 1) * n_s) * (eta_max - eta_min) / n_s

                j_start, j_end = j * self.batch_size, (j + 1) * self.batch_size
                X_batch = X[:, j_start:j_end]
                Y_batch = Y[:, j_start:j_end]
                grad_W_l, grad_b_l = self.backward_pass(X_batch, Y_batch)
                for idx, grad_W in enumerate(grad_W_l):
                    self.W[idx] += -learning_rate*grad_W
                    self.b[idx] += -learning_rate*grad_b_l[idx]
                if t >= 2 * num_cycle * n_s:
                    breakk = True
                    break
                # We reached the bottom after having been at the top of the cycle, so increase l
                if t % (2 * n_s) == 0:
                    l += 1
                t += 1
                eta_l.append(learning_rate)
            # Training loss
            t_c, t_l = self.compute_cost(X, Y)
            cost_training.append(t_c)
            loss_training.append(t_l)
            # Eval loss
            e_c, e_l = self.compute_cost(self.X_eval, self.Y_eval)
            cost_eval.append(e_c)
            loss_eval.append(e_l)
            # Accuracy
            acc_t.append(self.compute_acc(X, y, label='training'))
            acc_e.append(self.compute_acc(self.X_eval, self.y_eval, label='validation'))
        return cost_training, cost_eval, loss_training, loss_eval, acc_t, acc_e, eta_l

    def evaluate(self, X_test, y_test):
        self.compute_acc(X_test, y_test, label='testing', verbose=True)

    def compare_gradients(self, g_a, g_n, eps=1e-6):
        # is this value small? then good, we have almost the same gradient.
        assert (g_a.shape == g_n.shape), print('<| shape of g_a: {}\n<| shape of g_n: {}'.format(g_a.shape, g_n.shape))
        err = np.zeros(g_a.shape)
        for i in range(err.shape[0]):
            for j in range(err.shape[1]):
                err[i, j] = np.abs(g_a[i, j] - g_n[i, j]) / max(eps, np.abs(g_a[i, j]) + np.abs(g_n[i, j]))

        if self.verbose:
            print('<| analytical gradient is: \n', g_a)
            print('<| numerical gradient is: \n', g_n)

        return err

    def __del__(self):
        """
        func __del__/1
        @spec (self) :: None
            Performs whatever is in this method whenever an instance of the class is about to be deleted.
        """
        print('<| Deleting instance of class KNN ...')

    def __repr__(self):
        """
        func __repr__/1
        @spec (self) :: str
            This method is called whenever the repr() method is called, simply returns a string with the
            current instance and representation of the class object KNN.
        """
        return f'<| KNN with parameters:\n\t\tbatch_size={self.batch_size}\n\t\tn_epochs={self.n_epochs}\n\t\t' \
               f'learning_rate={self.eta}\n\t\tlambda={self.lamda}\n\t\tnum_layers={self.num_layers}\n\t\t' \
               f'num_nodes={self.num_nodes}\n\t\tverbose={self.verbose}'

    def __format__(self, format_spec):
        """
        func __format__/2
        @spec (self, str) :: str
            I actually don't really know what this method is for. 'Called to compute a formatted string of x'.
        """
        if format_spec == 'f':
            return super().__repr__()
        return str(self)

    @staticmethod
    def plot(training_loss, eval_loss, label):
        plt.plot([i for i in range(len(training_loss))], training_loss, color='green', linewidth=2, label='training')
        if eval_loss:
            plt.plot([i for i in range(len(eval_loss))], eval_loss, color='red', linewidth=2, label='validation')
        plt.xlabel('epoch')
        plt.ylabel(label)
        plt.legend()
        plt.show()


def main():
    # -- Unit testing --
    np.random.seed(69)
    knn = KNN(batch_size=100, n_epochs=20, eta=1e-3, lamda=5e-3,
              num_layers=9, num_nodes=[50, 30, 20, 20, 10, 10, 10, 10], dimensionality=3072,
              shuffle=True, batch_norm=False, verbose=True)
    knn.parse_full_data(val_split=5000)
    knn.initialize_params()
    P, S_l, S_norm_l, H, mu, std = knn.forward_pass_BN()
    """
    gradwl, gradbl = knn.backward_pass(knn.X[:, :2], knn.Y[:, :2])
    numwl, numbl, _, _ = knn.compute_grads_num_slow(knn.X[:, :2], knn.Y[:, :2])
    print(knn.compare_gradients(g_a=gradbl[2], g_n=numbl[2]))
    cost_training, cost_eval, loss_training, loss_eval, acc_t, acc_e, eta_l = knn.fit(knn.X, knn.Y, knn.y)
    knn.plot(cost_training, cost_eval, 'cost')
    knn.plot(loss_training, loss_eval, 'loss')
    knn.plot(acc_t, acc_e, 'accuracy')
    knn.plot(eta_l, [], 'learning rate')
    knn.evaluate(knn.X_test, knn.y_test)
    """


if __name__ == '__main__':
    main()
