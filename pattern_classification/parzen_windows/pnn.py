import pandas as pd
import numpy as np

def get_data():
    """
     Extract data from csv
    :return: Tuple of 2-d Numpy Arrays
    """
    # d = pd.read_csv('C:\\Users\\Justin\\PycharmProjects\\machine_learning\\pattern_classification\\hw3_2_1.csv', header=None).transpose().values
    d1 = pd.read_csv('hw3_2_1.csv', header=None).transpose().values
    d2 = pd.read_csv('hw3_2_2.csv', header=None).transpose().values
    return d1, d2


def normalize_patterns(train_x):
    """
    Square of the features should sum to 1 for each sample
    n - rows (number of samples)
    d - cols (number of features)
    :param train_x: np.array (n x d) - training data
    :return:        np.array (n x d) - Normalized training data
    """
    x = np.zeros((len(train_x), 2))
    s = np.sqrt(np.square(train_x[:,0]) + np.square(train_x[:,1]))
    for col in range(train_x.shape[1]):
        x[:,col] = train_x[:,col] / s
    return x


def compute_weight_links(d):
    """ Trivial for PNN"""
    return d


def pnn_train(d):
    """
    Train the probablistic neural network
    :param d: np.array(n x d) - training data for 1 class
    :return:  np.array(n x d) - weighted links for same class
    """
    x = normalize_patterns(d)
    w = compute_weight_links(x)
    return w


def normalize_test_pattern(x):
    """
    Same normalization technique but for a test pattern
    :param x: np.array(1 x d) - test sample
    :return:  np.array(1 x d) - test sample
    """
    return x / np.sqrt(np.square(x[0]) + np.square(x[1]))


def compute_net_activation(w, x):
    """
    Net Activation function is the dot product between the weighted links and the test sample
    :param w: np.array(n x d) - weighted links for a class
    :param x: np.array(1 x d) - test sample
    :return: np.array(n x 1)  - our pnn
    """
    return np.sum(w * x, axis=1)


def activate_function(net, gww):
    """
    Takes the sum of output units for each activation function
    :param net: np.array(n x 1) - probabilistic neural net for 1 class
    :param gww:           float - Gaussian window width
    :return:              float - output value for a class
    """
    return np.sum(np.exp((net - 1) / gww**2))


def pnn_classify(x, w1, w2, gww):
    """
    Classifies a training sample using a probabilistic neural network
    :param x:  np.array(1 x d) - test sample
    :param w1: np.array(n x d) - weighted links for class 1
    :param w2: np.array(n x d) - weighted links for class 2
    :param gww:          float - Gaussian window width
    :return:
    """
    n_x = normalize_test_pattern(x)
    net1 = compute_net_activation(w1, n_x)
    net2 = compute_net_activation(w2, n_x)

    g1 = activate_function(net1, gww)
    g2 = activate_function(net2, gww)
    print('Total output unit for Class 1: {0:0.03f} and Class 2: {1:0.03f}'.format(g1, g2))
    if g1 > g2:
        print('Test Sample:{} belongs to Class 1'.format(x))
    else:
        print('Test Sample:{} belongs to Class 2'.format(x))


d1, d2 = get_data()
x1 = normalize_patterns(d1)
print('Check Squared total is 1 -> {0:0.5f} + {1:0.5f} = {2}'.format(x1[0,0]**2, x1[0,1]**2, x1[0,0]**2 + x1[0,1]**2))

gww = 0.2
w1 = pnn_train(d1)
w2 = pnn_train(d2)

test_pattern = np.array([1, -2])
pnn_classify(test_pattern, w1, w2, gww)