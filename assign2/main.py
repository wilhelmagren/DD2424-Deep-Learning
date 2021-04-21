"""
Author: Wilhelm Ågren, wagren@kth.se
Last edited: 21/04/2021
"""

import numpy as np
from tqdm import tqdm
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
def preprocess_data(x, xx):
    print('<| Preprocess X data :')
    train_variance = x.std(axis=1).reshape(x.shape[0], 1)
    train_mean = x.mean(axis=1).reshape(x.shape[0], 1)
    normalized = (xx - train_mean) / train_variance
    return normalized


# Third step, initialize the parameters of the model W and b since we now dimensionality's
def initialize_params(K, d, m, verbose=False):
    print('<| Initialize params :')
    # W1: m x d
    # W2: K x m
    # b1: m x 1
    # b2: K x 1
    W1 = np.random.normal(0, 1 / np.sqrt(d), size=(m, d))
    W2 = np.random.normal(0, 1 / np.sqrt(m), size=(K, m))
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
        s1 = W1 @ X + b1
    else:
        tmp = W1 @ X
        s1 = np.asmatrix(tmp).T + b1
    h = np.maximum(0, s1)
    s = W2 @ h + b2
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
    regularization_sum = lamb * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
    loss_sum = 0
    for col in range(num_points):
        loss_sum += l_cross(X[:, col], Y[:, col], W1, W2, b1, b2)
    cost = loss_sum / num_points + regularization_sum
    # Cost becomes a 1 x 1 matrix but J is supposed to be a scalar, so return only the scalar value
    return cost[0, 0], (loss_sum / num_points)[0, 0]


def l_cross(x, y, W1, W2, b1, b2):
    P, _ = evaluate_classifier(x, W1, W2, b1, b2)
    return -np.log(np.dot(y.T, P))


# Sixth step, write a function that computes the accuracy of the network's predictions given by equation 4.
def compute_accuracy(X, y, W1, W2, b1, b2, verbose=False):
    print('<| Compute accuracy :') if verbose else None

    total = y.shape[0]
    correct = 0
    P, _ = evaluate_classifier(X, W1, W2, b1, b2)
    for k in range(total):
        pred = np.argmax(P[:, k])
        if pred == y[k]:
            correct += 1

    return correct / total


# Seventh step, write the function that evaluates, for a mini-batch, the gradients of the cost function w.r.t W and b
def compute_gradients(X, Y, W1, W2, b1, b2, lamb, verbose=False):
    print('<| Compute gradient analytically [fast]:') if verbose else None
    """
    Each column of X corresponds to an image, and X has size d x n.
    Each column of Y is the one-hot ground truth label for the corresponding column of X, and Y has size K x n.
    Each column of P contains the probability for each label for the image in the corresponding column of X,
        and P has size K x n.
    """
    # Forward pass
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
    grad_W2 = 2 * lamb * W2 + g_batch @ H.T / len_D

    # 3. Propagate the gradient back through the second layer
    g_batch = W2.T @ g_batch
    g_batch = np.multiply(g_batch, H > 0)

    # 4. Then
    grad_W1 = 2 * lamb * W1 + (g_batch @ X.T) / len_D
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
    assert (g_a.shape == g_n.shape), print('<| shape of g_a: {}\n<| shape of g_n: {}'.format(g_a.shape, g_n.shape))
    err = np.zeros(g_a.shape)
    for i in range(err.shape[0]):
        for j in range(err.shape[1]):
            err[i, j] = np.abs(g_a[i, j] - g_n[i, j]) / max(eps, np.abs(g_a[i, j]) + np.abs(g_n[i, j]))

    if verbose:
        print('<| analytical gradient is: ', g_a)
        print('<| numerical gradient is: ', g_n)

    return err


# Eighth step, perform the mini-batch gradient descent algorithm to learn the network's parameters.
# Implement vanilla batch learning, no adaptive tuning of the learning parameter or momentum terms.
def minibatch_GD(X, Y, GDparams, W1, W2, b1, b2, lamb, eval_X, eval_Y, num_cycle, verbose=False):
    cost_training, loss_training, cost_eval, loss_eval, acc_t, eta_l, acc_e = [], [], [], [], [], [], []
    eta_min, eta_max, n_s, l, t, learning_rate = 1e-5, 1e-1, 500, 0, 1, GDparams['eta']
    n_batch, n_epochs = GDparams['n_batch'], GDparams['n_epochs']
    breakk = False
    assert (X.shape[1] % n_batch == 0), print('<| CAN NOT SPLIT DATA ACCORDINGLY')
    # ------------------------------------------------------------------------------------------------------------------
    # Training loss
    t_c, t_l = compute_cost(X, Y, W1, W2, b1, b2, lamb)
    cost_training.append(t_c)
    loss_training.append(t_l)
    # Eval loss
    e_c, e_l = compute_cost(eval_X, eval_Y, W1, W2, b1, b2, lamb)
    cost_eval.append(e_c)
    loss_eval.append(e_l)
    # Accuracy
    acc_t.append(compute_accuracy(X, y, W1, W2, b1, b2))
    acc_e.append(compute_accuracy(eval_X, eval_y, W1, W2, b1, b2))
    # ------------------------------------------------------------------------------------------------------------------
    for _ in tqdm(range(n_epochs)):
        if breakk:
            break
        for j in range(int(X.shape[1] / n_batch)):
            if 2 * l * n_s <= t <= (2 * l + 1) * n_s:
                # Learning increasing
                learning_rate = eta_min + (eta_max - eta_min) * (t - 2 * l * n_s) / n_s
            elif (2 * l + 1) * n_s <= t <= 2 * (l + 1) * n_s:
                # Learning rate decreasing
                learning_rate = eta_max - (t - (2 * l + 1) * n_s) * (eta_max - eta_min) / n_s

            j_start, j_end = j * n_batch, (j + 1) * n_batch
            print('<| \tmini-batch on index [{}, {}]'.format(j_start, j_end)) if verbose else None
            X_batch = X[:, j_start:j_end]
            Y_batch = Y[:, j_start:j_end]
            grad_W1, grad_W2, grad_b1, grad_b2 = compute_gradients(X_batch, Y_batch, W1, W2, b1, b2, lamb)
            W1 += -learning_rate * grad_W1
            W2 += -learning_rate * grad_W2
            b1 += -learning_rate * grad_b1
            b2 += -learning_rate * grad_b2
            if t >= 2 * num_cycle * n_s:
                breakk = True
                break
            # We reached the bottom after having been at the top of the cycle, so increase l
            if t % (2 * n_s) == 0:
                l += 1
            t += 1
            eta_l.append(learning_rate)
        # Training loss
        t_c, t_l = compute_cost(X, Y, W1, W2, b1, b2, lamb)
        cost_training.append(t_c)
        loss_training.append(t_l)
        # Eval loss
        e_c, e_l = compute_cost(eval_X, eval_Y, W1, W2, b1, b2, lamb)
        cost_eval.append(e_c)
        loss_eval.append(e_l)
        # Accuracy
        acc_t.append(compute_accuracy(X, y, W1, W2, b1, b2))
        acc_e.append(compute_accuracy(eval_X, eval_y, W1, W2, b1, b2))
    return W1, W2, b1, b2, cost_training, cost_eval, loss_training, loss_eval, acc_t, acc_e, eta_l


# Given from func minibatch_GD/6 is the list of the cost after each epoch training. Visualize it.
def plot_loss(training_loss, eval_loss):
    plt.plot([i for i in range(len(training_loss))], training_loss, color='green', linewidth=2, label='training loss')
    plt.plot([i for i in range(len(eval_loss))], eval_loss, color='red', linewidth=2, label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def plot_cost(training_loss, eval_loss):
    plt.plot([i for i in range(len(training_loss))], training_loss, color='green', linewidth=2, label='training')
    plt.plot([i for i in range(len(eval_loss))], eval_loss, color='red', linewidth=2, label='validation')
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.legend()
    plt.show()


def plot_acc(acc_t, acc_e):
    plt.plot([i for i in range(len(acc_t))], acc_t, color='green', linewidth=2, label='training')
    plt.plot([i for i in range(len(acc_e))], acc_e, color='red', linewidth=2, label='validation')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


def plot_eta(eta_l):
    plt.plot([i for i in range(len(eta_l))], eta_l, color='blue', linewidth=2)
    plt.xlabel('update step')
    plt.ylabel('learning rate')
    plt.show()


########################################################################################################################
def compute_grads_slow(X, Y, W1, W2, b1, b2, lamb, verbose=False, h=1e-5):
    # Initialize gradients to correct shape
    grad_W1 = np.zeros(shape=W1.shape)
    grad_W2 = np.zeros(shape=W2.shape)
    grad_b1 = np.zeros(shape=b1.shape)
    grad_b2 = np.zeros(shape=b2.shape)

    # Compute b1 gradients
    for i in range(b1.shape[0]):
        try_b = b1
        try_b[i] -= h
        c1, _ = compute_cost(X, Y, W1, W2, try_b, b2, lamb)

        try_b = b1
        try_b[i] += h
        c2, _ = compute_cost(X, Y, W1, W2, try_b, b2, lamb)
        grad_b1[i] = (c2 - c1) / (2 * h)

    # Compute b2 gradients
    for i in range(b2.shape[0]):
        try_b = b2
        try_b[i] -= h
        c1, _ = compute_cost(X, Y, W1, W2, b1, try_b, lamb)

        try_b = b2
        try_b[i] += h
        c2, _ = compute_cost(X, Y, W1, W2, b1, try_b, lamb)

        grad_b2[i] = (c2 - c1) / (2 * h)

    # Compute W1 gradients
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            try_W = W1
            try_W[i, j] -= h
            c1, _ = compute_cost(X, Y, try_W, W2, b1, b2, lamb)

            try_W = W1
            try_W[i, j] += h
            c2, _ = compute_cost(X, Y, try_W, W2, b1, b2, lamb)

            grad_W1[i, j] = (c2 - c1) / (2 * h)

    # Compute W2 gradients
    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            try_W = W2
            try_W[i, j] -= h
            c1, _ = compute_cost(X, Y, W1, try_W, b1, b2, lamb)

            try_W = W2
            try_W[i, j] += h
            c2, _ = compute_cost(X, Y, W1, try_W, b1, b2, lamb)

            grad_W2[i, j] = (c2 - c1) / (2 * h)

    if verbose:
        print('<| grad_W1: ', grad_W1)
        print('<| grad_W2: ', grad_W2)
        print('<| grad_b1: ', grad_b1)
        print('<| grad_b2: ', grad_b2)

    assert grad_W1.shape == W1.shape and grad_W2.shape == W2.shape \
           and grad_b1.shape == b1.shape and grad_b2.shape == b2.shape, \
           print('<| Incorrect gradient shape, terminating...')

    return grad_W1, grad_W2, grad_b1, grad_b2


def compute_grads(X, Y, W1, W2, b1, b2, lamb, verbose=False, h=1e-5):
    W_l, b_l = [W1, W2], [b1, b2]
    grad_W_l, grad_b_l = [0, 0], [0, 0]
    c, _ = compute_cost(X, Y, W1, W2, b1, b2, lamb)

    for j in range(len(b_l)):
        grad_b_l[j] = np.zeros(shape=b_l[j].shape)
        for i in range(len(b_l[j])):
            b_try = b_l
            b_try[j][i] = b_try[j][i] + h
            c2, _ = compute_cost(X, Y, W_l[0], W_l[1], b_try[0], b_try[1], lamb)
            grad_b_l[j][i] = (c2 - c) / h

    for j in range(len(W_l)):
        grad_W_l[j] = np.zeros(shape=W_l[j].shape)
        for i in range(len(W_l[j])):
            W_try = W_l
            W_try[j][i] = W_try[j][i] + h
            c2, _ = compute_cost(X, Y, W_try[0], W_try[1], b_l[0], b_l[1], lamb)
            grad_W_l[j][i] = (c2 - c) / h
    if verbose:
        print('<| grad_W1: ', grad_W_l[0])
        print('<| grad_W2: ', grad_W_l[1])
        print('<| grad_b1: ', grad_b_l[0])
        print('<| grad_b2: ', grad_b_l[1])

    return grad_W_l[0], grad_W_l[1], grad_b_l[0], grad_b_l[1]
########################################################################################################################


def get_small_batches():
    dataX, dataY, datay = parse_data(loadBatch('data_batch_1'))
    eval_X, eval_Y, eval_y = parse_data(loadBatch('data_batch_2'))
    test_X, test_Y, test_y = parse_data(loadBatch('test_batch'))

    train_variance = dataX.std(axis=1).reshape(dataX.shape[0], 1)
    train_mean = dataX.mean(axis=1).reshape(dataX.shape[0], 1)

    dataX = (dataX - train_mean)/train_variance
    eval_X = (eval_X - train_mean) / train_variance
    test_X = (test_X - train_mean) / train_variance

    return dataX, dataY, datay, eval_X, eval_Y, eval_y, test_X, test_Y, test_y


def get_big_batches():
    dataX, dataY, datay = parse_data(loadBatch('data_batch_1'))
    dataX2, dataY2, datay2 = parse_data(loadBatch('data_batch_2'))
    dataX3, dataY3, datay3 = parse_data(loadBatch('data_batch_3'))
    dataX4, dataY4, datay4 = parse_data(loadBatch('data_batch_4'))
    dataX5, dataY5, datay5 = parse_data(loadBatch('data_batch_5'))
    X, Y, y = np.concatenate((dataX, dataX2, dataX3, dataX4, dataX5[:, :9000]), axis=1), \
              np.concatenate((dataY, dataY2, dataY3, dataY4, dataY5[:, :9000]), axis=1), \
              np.concatenate((datay, datay2, datay3, datay4, datay5[:9000]))
    eval_X, eval_Y, eval_y = dataX5[:, 9000:], dataY5[:, 9000:], datay5[9000:]
    test_X, test_Y, test_y = parse_data(loadBatch('test_batch'))
    train_variance = X.std(axis=1).reshape(X.shape[0], 1)
    train_mean = X.mean(axis=1).reshape(X.shape[0], 1)

    X = (X - train_mean)/train_variance
    eval_X = (eval_X - train_mean)/train_variance
    test_X = (test_X - train_mean)/train_variance

    return X, Y, y, eval_X, eval_Y, eval_y, test_X, test_Y, test_y


if __name__ == '__main__':
    X, Y, y, eval_X, eval_Y, eval_y, test_X, test_Y, test_y = get_small_batches()
    hid_nodes = 50  # m
    # PERFORM GRID SEARCH THROUGH HYPERPARAMETER SPACE TO FIND LAMBDA
    # dom = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    # dom = [1e-5, 3e-5, 5e-5, 8e-5, 1e-4, 3e-4, 5e-4, 8e-4, 1e-3, 3e-3, 5e-3, 8e-3, 1e-2, 3e-2, 5e-2]
    filename = 'valid_scores_fine_3cycle.txt'
    W1, W2, b1, b2 = initialize_params(Y.shape[0], X.shape[0], hid_nodes, True)
    # Good lambda 5e-3
    batch_n, lamb = 100, 0.01
    learning_rate, n_epochs, num_cyc = 0.001, 10, 1
    params = {'n_batch': batch_n, 'eta': learning_rate, 'n_epochs': n_epochs}
    # """
    W1_upd, W2_upd, b1_upd, b2_upd, cost_t, cost_e, loss_t, loss_e, acc_t, acc_e, eta_l = minibatch_GD(X, Y, params, W1,
                                                                                                       W2, b1, b2, lamb,
                                                                                                       eval_X, eval_Y,
                                                                                                       num_cyc)
    plot_loss(loss_t, loss_e)
    plot_eta(eta_l)
    plot_acc(acc_t, acc_e)
    plot_cost(cost_t, cost_e)
    #test_X, test_Y, test_y = parse_data(loadBatch('test_batch'), True)
    #test_X = preprocess_data(test_X)
    test_acc = compute_accuracy(test_X, test_y, W1_upd, W2_upd, b1_upd, b2_upd)
    print('<| Final test acc: [{}]'.format(test_acc))
    """ FOR WRITING TO FILE DURING GRID SEARCH OF LAMBDA
    f = open(filename, 'a')
    f.write('<| | | | | | | | | | | |\nlambda={}\n'.format(lambd))
    f.write('Validation loss: ' + str(min(loss_e)) + '\n')
    f.write('Validation acc: ' + str(max(acc_e)) + '\n')
    f.write('Test acc: ' + str(test_acc) + '\n\n')
    f.close()
    """
