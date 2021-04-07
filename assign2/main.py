"""
Author: Wilhelm Ågren, wagren@kth.se
Last edited: 31/3/2021
"""

import numpy as np
import matplotlib.pyplot as plt


def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def loadBatch(filename):
    """ Copied from the dataset website """
    import pickle
    with open('Dataset/' + filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# First step, read the data from the CIFAR-10 batch file and format it
def parse_data(dset, verbose=False):
    print('<| Parse data from batch :')
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


# Second step, now preprocess the raw input data such that it helps training.
def preprocess_data(x):
    print('<| Preprocess X data :')
    normalized = (x - x.mean(axis=0)) / x.std(axis=0)
    return normalized


# Third step, initialize the parameters of the model W and b since we now dimensionality's
def initialize_params(K, d, m, verbose=False):
    print('<| Initialize params :')
    # W1: m x d
    # W2: K x m
    # b1: m x 1
    # b2: K x 1
    W1 = np.random.normal(0, 1/np.sqrt(d), size=(m, d))
    W2 = np.random.normal(0, 1/np.sqrt(m), size=(K, m))
    b1 = np.zeros(shape=(m, 1))
    b2 = np.zeros(shape=(K, 1))

    if verbose:
        print('\tthe shape of W1:', W1.shape)
        print('\tthe shape of W2:', W2.shape)
        print('\tthe shape of b1:', b1.shape)
        print('\tthe shape of b2:', b2.shape)

    return W1, W2, b1, b2


# Fourth step, write a function that evaluates the network function,
# i.e. equations 1, 2 (see notes) on multiple images and returns the results.
def evaluate_classifier(X, W1, W2, b1, b2, verbose=False):
    s1 = 0
    if len(X.shape) > 1:
        s1 = W1@X + b1
    else:
        tmp = W1@X
        s1 = np.asmatrix(tmp).T + b1
    h = np.maximum(0, s1)
    s = W2@h + b2
    p = softmax(s)

    if verbose:
        print('\ts becomes:', s)
        print('\tsoftmax(s) = p yields:', p)

    return p, h


# Fifth step, write the function that computes the cost function given the equation 5 (see notes),
# for a set of images. Lambda is the regularization term.
def compute_cost(X, Y, W1, W2, b1, b2, lamb):
    """
    1/D SUM_[x,y in D] l_cross(x,y,W,b) + lambda SUM_[i,j] W_[i,j]^2
    """
    if len(X.shape) < 2:
        X = np.asmatrix(X).T
    if len(Y.shape) < 2:
        Y = np.asmatrix(Y).T
    assert (X.shape[1] == Y.shape[1])

    num_points = X.shape[1]
    regularization_sum = lamb * (np.sum(W1**2) + np.sum(W2**2))
    loss_sum = 0
    for col in range(num_points):
        loss_sum += l_cross(X[:, col], Y[:, col], W1, W2, b1, b2)
    cost = loss_sum/num_points + regularization_sum
    # Cost becomes a 1 x 1 matrix but J is supposed to be a scalar, so return only the scalar value
    return cost[0, 0], loss_sum/num_points


def l_cross(x, y, W1, W2, b1, b2):
    P, _ = evaluate_classifier(x, W1, W2, b1, b2)
    return -np.log(np.dot(y.T, P))


# Sixth step, write a function that computes the accuracy of the network's predictions given by equation 4.
def compute_accuracy(X, y, W, b):
    print('<| Compute accuracy :')
    total = y.shape[0]
    correct = 0
    P = evaluate_classifier(X, W, b)
    for k in range(total):
        pred = np.argmax(P[:, k])
        if pred == y[k]:
            correct += 1
    return correct/total


# Seventh step, write the function that evaluates, for a mini-batch, the gradients of the cost function w.r.t W and b
def compute_gradients(X, Y, W1, W2, b1, b2, lamb, verbose=False):
    print('<| Compute gradient analytically [fast]:') if verbose else None
    """
    Each column of X corresponds to an image, and X has size d x n.
    Each column of Y is the one-hot ground truth label for the corresponding column of X, and Y has size K x n.
    Each column of P contains the probability for each label for the image in the corresponding column of X,
        and P has size K x n.
    """
    P, H = evaluate_classifier(X, W1, W2, b1, b2)

    if len(X.shape) < 2:
        X = np.asmatrix(X).T
    len_D = X.shape[1]
    if len(Y.shape) < 2:
        Y = np.asmatrix(Y).T

    # 1. Set
    g_batch = -(Y - P)

    # 2. Then
    grad_b2 = np.sum(g_batch / len_D, axis=1)
    if len(grad_b2.shape) < 2:
        grad_b2 = np.asmatrix(grad_b2).T
    grad_W2 = 2*lamb*W2 + g_batch@H.T / len_D

    # 3. Propagate the gradient back through the second layer
    g_batch = W2.T@g_batch
    g_batch = np.multiply(g_batch, H > 0)

    # 4. Then
    grad_W1 = 2*lamb*W1 + (g_batch@X.T) / len_D
    grad_b1 = np.sum(g_batch / len_D, axis=1)
    if len(grad_b1.shape) < 2:
        grad_b1 = np.asmatrix(grad_b1).T

    assert grad_W1.shape == (50, 3072)
    assert grad_W2.shape == (10, 50)
    assert grad_b1.shape == (50, 1)
    assert grad_b2.shape == (10, 1)

    return grad_W1, grad_W2, grad_b1, grad_b2


# You should compare your analytical gradient with the numerical gradient by examining their
# absolute differences and declaring if all these absolute differences are small.
# A more reliable method is to compute the relative error between a numerically computed gradient value g_n
# and an analytically computed gradient value g_a
def compare_gradients(g_a, g_n, verbose=False, eps=1e-6):
    # is this value small? then good, we have almost the same gradient.
    assert(g_a.shape == g_n.shape), print('<| shape of g_a: {}\n<| shape of g_n: {}'.format(g_a.shape, g_n.shape))
    err = np.zeros(g_a.shape)
    for i in range(err.shape[0]):
        for j in range(err.shape[1]):
            err[i, j] = np.abs(g_a[i, j] - g_n[i, j]) / max(eps, np.abs(g_a[i, j]) + np.abs(g_n[i, j]))

    if verbose:
        print('<| Compare gradients:\n', err)

    return err


# Eighth step, perform the mini-batch gradient descent algorithm to learn the network's parameters.
# Implement vanilla batch learning, no adaptive tuning of the learning parameter or momentum terms.
def minibatch_GD(X, Y, GDparams, W, b, lamb, eval_X, eval_Y, verbose=False):
    cost_training, cost_eval = [], []
    n_batch, n_epochs = GDparams['n_batch'], GDparams['n_epochs']
    assert(X.shape[1] % n_batch == 0), print('<| CAN NOT SPLIT DATA ACCORDINGLY')
    for i in range(n_epochs):
        print('<| Epoch [{}]'.format(i + 1))
        for j in range(int(X.shape[1] / n_batch)):
            j_start, j_end = j * n_batch, (j+1) * n_batch
            print('<| \tmini-batch on index [{}, {}]'.format(j_start, j_end)) if verbose else None
            X_batch = X[:, j_start:j_end]
            Y_batch = Y[:, j_start:j_end]
            P = evaluate_classifier(X[:, j_start:j_end], W, b)
            grad_W, grad_b = compute_gradients(X_batch, Y_batch, P, W, lamb)
            W += -learning_rate * grad_W
            b += -learning_rate * grad_b
        # Training loss
        cost_training.append(compute_cost(X, Y, W, b, lamb))
        # Eval loss
        cost_eval.append(compute_cost(eval_X, eval_Y, W, b, lamb))
    return W, b, cost_training, cost_eval


# Given from func minibatch_GD/6 is the list of the cost after each epoch training. Visualize it.
def plot_loss(training_loss, eval_loss):
    plt.plot([i for i in range(len(training_loss))], training_loss, color='green', linewidth=2, label='training loss')
    plt.plot([i for i in range(len(eval_loss))], eval_loss, color='red', linewidth=2, label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def compute_grads_slow(X, Y, W1, W2, b1, b2, lamb, h=1e-4):
    W_l, b_l = [W1, W2], [b1, b2]
    grad_W_l, grad_b_l = [0, 0], [0, 0]
    for j in range(len(b_l)):
        grad_b_l[j] = np.zeros(shape=b_l[j].shape)
        for i in range(len(b_l[j])):
            b_try = b_l
            b_try[j][i] = b_try[j][i] - h
            c1, _ = compute_cost(X, Y, W_l[0], W_l[1], b_try[0], b_try[1], lamb)

            b_try = b_l
            b_try[j][i] = b_try[j][i] + h
            c2, _ = compute_cost(X, Y, W_l[0], W_l[1], b_try[0], b_try[1], lamb)

            grad_b_l[j][i] = (c2 - c1) / (2*h)

    for j in range(len(W_l)):
        grad_W_l[j] = np.zeros(shape=W_l[j].shape)
        for i in range(len(W_l[j])):
            W_try = W_l
            W_try[j][i] = W_try[j][i] - h
            c1, _ = compute_cost(X, Y, W_try[0], W_try[1], b_l[0], b_l[1], lamb)

            W_try = W_l
            W_try[j][i] = W_try[j][i] + h
            c2, _ = compute_cost(X, Y, W_try[0], W_try[1], b_l[0], b_l[1], lamb)

            grad_W_l[j][i] = (c2 - c1) / (2*h)

    return grad_W_l[0], grad_W_l[1], grad_b_l[0], grad_b_l[1]


if __name__ == '__main__':
    # http://www.cs.toronto.edu/~kriz/cifar.html
    data = loadBatch('data_batch_1')
    np.random.seed(69)
    X, Y, y = parse_data(data, True)
    eval_X, eval_Y, eval_y = parse_data(loadBatch('data_batch_2'), True)
    X, eval_X = preprocess_data(X), preprocess_data(eval_X)
    hid_nodes = 50  # m
    W1, W2, b1, b2 = initialize_params(Y.shape[0], X.shape[0], hid_nodes, True)
    batch_n, lamb = 20, 0
    learning_rate, n_epochs = 0.001, 40
    params = {'n_batch': batch_n, 'eta': learning_rate, 'n_epochs': n_epochs}
    grad_W1, grad_W2, grad_b1, grad_b2 = compute_gradients(X[:, :batch_n], Y[:, :batch_n], W1, W2, b1, b2, lamb)
    gw1, gw2, gb1, gb2 = compute_grads_slow(X[:, :batch_n], Y[:, :batch_n], W1, W2, b1, b2, lamb)
    print(compare_gradients(grad_b2, gb2))
