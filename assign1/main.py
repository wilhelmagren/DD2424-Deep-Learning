from functions import *
import numpy as np


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
def initialize_params(K, d, verbose=False):
    print('<| Initialize params :')
    W = np.random.normal(0, 0.01, size=(K, d))
    b = np.ones((K, 1))

    if verbose:
        print('\tthe shape of W:', W.shape)
        print('\tthe shape of b:', b.shape)

    return W, b


# Fourth step, write a function that evaluates the network function,
# i.e. equations 1, 2 (see notes) on multiple images and returns the results.
def evaluate_classifier(X, W, b, verbose=False):
    s = 0
    if len(X.shape) > 1:
        s = W@X + b
    else:
        tmp = W@X
        s = np.asmatrix(tmp).T + b
    p = softmax(s)

    if verbose:
        print('\ts becomes:', s)
        print('\tsoftmax(s) = p yields:', p)

    return p


# Fifth step, write the function that computes the cost function given the equation 5 (see notes),
# for a set of images. Lambda is the regularization term.
def compute_cost(X, Y, W, b, lamb=0.1):
    print('<| Compute cost :')
    """
    1/D SUM_[x,y in D] l_cross(x,y,W,b) + lambda SUM_[i,j] W_[i,j]^2
    """
    if len(X.shape) < 2:
        X = np.asmatrix(X).T
    if len(Y.shape) < 2:
        Y = np.asmatrix(Y).T
    assert (X.shape[1] == Y.shape[1])
    num_points = X.shape[1]
    regularization_sum = lamb * np.sum(W**2)
    loss_sum = 0
    for col in range(num_points):
        loss_sum += l_cross(X[:, col], Y[:, col], W, b)
    cost = (loss_sum + regularization_sum) / num_points
    # Cost becomes a 1 x 1 matrix but J is supposed to be a scalar, so return only the scalar value
    return cost[0, 0]


def l_cross(x, y, W, b):
    P = evaluate_classifier(x, W, b)
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
def compute_gradients(X, Y, P, W, lamb):
    """
    Each column of X corresponds to an image, and X has size d x n.
    Each column of Y is the one-hot ground truth label for the corresponding column of X, and Y has size K x n.
    Each column of P contains the probability for each label for the image in the corresponding column of X,
        and P has size K x n.
    grad_W is the gradient matrix of the cost J relative to W and has size K x d.
    grad_b is the gradient vector of the cost J relative to b and has size K x 1.
    """
    if len(X.shape) < 2:
        X = np.asmatrix(X).T
    len_D = X.shape[1]
    if len(Y.shape) < 2:
        Y = np.asmatrix(Y).T
    # Equation 11, which calculates the gradient w.r.t b
    # 1. Evaluate P = softmax(Wx + b)
    # 2. let g = -(y-p)^T
    # 3. Add gradient of l(x, y, W, b) w.r.t b
    #           dL/db += g
    g_batch = -(Y - P)
    grad_b = g_batch / len_D
    grad_W = 2*lamb*W + (g_batch@X.T) / len_D
    return grad_W, grad_b


# Use the 'functions.py' grad func to compare by calculations,
def computeGradsNum(X, Y, P, W, b, lamda, h):
    """ Converted from matlab code """
    no = W.shape[0]
    d = X.shape[0]

    grad_W = np.zeros(W.shape);
    grad_b = np.zeros((no, 1));

    c = compute_cost(X, Y, W, b, lamda);

    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] += h
        c2 = compute_cost(X, Y, W, b_try, lamda)
        grad_b[i] = (c2 - c) / h
    """
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i, j] += h
            c2 = compute_cost(X, Y, W_try, b, lamda)
            grad_W[i, j] = (c2 - c) / h
    """
    return [grad_W, grad_b]


# You should compare your analytical gradient with the numerical gradient by examining their
# absolute differences and declaring if all these absolute differences are small.
# A more reliable method is to compute the relative error between a numerically computed gradient value g_n
# and an analytically computed gradient value g_a
def compare_gradients(g_a, g_n, eps=1e-2):
    # is this value small? then good, we have almost the same gradient.
    assert(g_a.shape == g_n.shape)
    err = np.zeros(g_a.shape)
    for i in range(err.shape[0]):
        for j in range(err.shape[1]):
            err[i, j] = np.abs(g_a[i, j] - g_n[i, j]) / max(eps, np.abs(g_a[i, j]) + np.abs(g_n[i, j]))
    return err


if __name__ == '__main__':
    # http://www.cs.toronto.edu/~kriz/cifar.html
    data = loadBatch('data_batch_1')
    np.random.seed(69)
    X, Y, y = parse_data(data, True)
    X = preprocess_data(X)
    W, b = initialize_params(Y.shape[0], X.shape[0], True)
    J = compute_cost(X, Y, W, b)
    acc = compute_accuracy(X, y, W, b)
    print(acc)
    P = evaluate_classifier(X[:, 100], W, b)
    grad_W, grad_b = compute_gradients(X[:, 100], Y[:, 100], P, W, 0.1)
    # ALRIGHT GOOD! My computed gradients are very close to those of the given function.
    ttt = computeGradsNum(X[:, 100], Y[:, 100], P, W, b, 0.1, 1)
    eps = compare_gradients(grad_b, ttt[1])
    print('gradient diff: ', eps)
