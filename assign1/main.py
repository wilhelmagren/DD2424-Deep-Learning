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
    pass


if __name__ == '__main__':
    # http://www.cs.toronto.edu/~kriz/cifar.html
    data = loadBatch('data_batch_1')
    # montage(data[b'data'])
    X, Y, y = parse_data(data, True)
    X = preprocess_data(X)
    W, b = initialize_params(Y.shape[0], X.shape[0], True)
    J = compute_cost(X, Y, W, b)
    print(J)
