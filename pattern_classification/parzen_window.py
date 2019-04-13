import pandas as pd
import numpy as np
from decimal import Decimal
from functools import reduce


def get_data():
    """
     Extract data from csv
    :return: Tuple of 2-d Numpy Arrays
    """
    # d = pd.read_csv('C:\\Users\\Justin\\PycharmProjects\\machine_learning\\pattern_classification\\hw3_2_1.csv', header=None).transpose().values
    d1 = pd.read_csv('hw3_2_1.csv', header=None).transpose().values
    d2 = pd.read_csv('hw3_2_2.csv', header=None).transpose().values
    return d1, d2


def g_kernal(x, x_i, h_n):
    """
    Gaussian Kernel (Window Function)
    :param x:    np.array (2x1) - estimation point
    :param x_i:  np.array (2x1) - a training sample
    :param h_n:  float          - width window
    :return:     np.array (2x1) - result from kernal
    """
    u = (x - x_i) / h_n
    return (1 / np.sqrt(2*np.pi)) * np.exp(-((np.dot(u, u)) / 2))


def estimate_density(data, x, h_n):
    """
    For every sample, we estimate the density for a given point
    :param data: np.array (2xN) - our training data for 1 class
    :param x:    np.array (2x1) - estimation point
    :param h_n:  float          - width window
    :return:     float          - parameter estimation for a class
    """
    tot = reduce(lambda a, b: a+b, [(1 / h_n) * g_kernal(x, d, h_n) for d in data])
    return tot / len(data)


def compute_parzen_window(d1, d2, g, h_n):
    """
    Estimate Densities of our data using Parzen Window technique
    :param d1:    np.array (2xN) - dataset belonging to class w1
    :param d2:    np.array (2xN) - dataset belonging to class w2
    :param g :    list [float]   - set of window steps [-4.0, -3.9 ... , 3.9, 4.0]
    :param h_n:   float          - window width
    :return:      tuple(np.array)
    """
    px_1 = [estimate_density(d1, np.array([float(i), float(j)]), h_n) for i in g for j in g]
    px_2 = [estimate_density(d2, np.array([float(i), float(j)]), h_n) for i in g for j in g]
    return px_1, px_2


# Start & end index * 10 to handle an iterable as a floating point
w_start = -40
w_end = 81
g = [p / 10 for p in range(w_start, w_end)]

h1 = 2
d1, d2 = get_data()
n = len(d1)
window_width = h1 / np.sqrt(n)  # 0.2

p1, p2 = compute_parzen_window(d1, d2, g, window_width)
print(p1)
print(p2)
