import pandas as pd
import numpy as np
from functools import reduce
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings

# Some configuration
mpl.style.use('seaborn')
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
warnings.filterwarnings('ignore')


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
    :param x:    np.array (2 x 1) - estimation point
    :param x_i:  np.array (2 x 1) - a training sample
    :param h_n:  float            - width window
    :return:     np.array (2 x 1) - result from kernal
    """
    u = (x - x_i) / h_n
    return (1 / np.sqrt(2*np.pi)) * np.exp(-((np.dot(u, u)) / 2))


def estimate_density(data, x, h_n):
    """
    For every sample, we estimate the density for a given point
    :param data: np.array (n x 2) - our training data for 1 class
    :param x:    np.array (2 x 1) - estimation point
    :param h_n:  float            - width window
    :return:     float            - parameter estimation for a class
    """
    tot = reduce(lambda a, b: a+b, [(1 / h_n) * g_kernal(x, d, h_n) for d in data])
    return tot / len(data)


def compute_parzen_window(d1, d2, g, h_n):
    """
    Estimate Densities of our data using Parzen Window technique
    :param d1:    np.array (n x 2) - dataset belonging to class w1
    :param d2:    np.array (n x 2) - dataset belonging to class w2
    :param g :    list [float]     - set of window steps [-4.0, -3.9 ... , 7.9, 8.0]
    :param h_n:   float            - window width
    :return:      tuple(np.array)
    """
    px_1 = [estimate_density(d1, np.array([float(i), float(j)]), h_n) for i in g for j in g]
    px_2 = [estimate_density(d2, np.array([float(i), float(j)]), h_n) for i in g for j in g]
    return px_1, px_2



def plot_all(x, y, z1, z2):
    """
    Plots multiple 3-d representations of the parzen windows
    let n be total number of steps
    :param x: np.array  (n x n) - meshgrid
    :param y: np.array  (n x n) - meshgrid (transpose of x)
    :param z1: np.array (n x n) - class 1 Likelihood for meshgrid
    :param z2: np.array (n x n) - class 2 Likelihood for meshgrid
    """
    zt1 = z1.copy()
    zt2 = z2.copy()
    zt1[zt1 < 0.0001] = np.nan
    zt2[zt2 < 0.0001] = np.nan
    fig = plt.figure(figsize=(8, 8))

    ax1 = fig.add_subplot(2, 2, 1, projection='3d', xlabel='Feature 1', ylabel='Feature 2', title='Class 1')
    s1 = ax1.plot_surface(x, y, z1, rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0, antialiased=True)

    ax2 = fig.add_subplot(2, 2, 2, projection='3d', xlabel='Feature 1', ylabel='Feature 2', title='Class 2')
    s1 = ax2.plot_surface(x, y, z2, rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0, antialiased=True)

    ax3 = fig.add_subplot(2, 2, 3, projection='3d', xlabel='Feature 1', ylabel='Feature 2', title='Both Classes')
    s1 = ax3.plot_surface(x, y, z1 + z2, rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0, antialiased=True)

    ax4 = fig.add_subplot(2, 2, 4, projection='3d', xlabel='Feature 1', ylabel='Feature 2', title='Both Classes')
    ax4.plot_surface(x, y, zt1, rstride=1, cstride=1, vmin=np.nanmin(zt1), vmax=np.nanmax(zt1),
                     cmap=plt.cm.coolwarm, linewidth=0, antialiased=True)
    ax4.plot_surface(x, y, zt2, rstride=1, cstride=1, vmin=np.nanmin(z2), vmax=np.nanmax(zt2),
                     cmap=plt.cm.Spectral, linewidth=0, antialiased=True)

    plt.show()


def train_parzen_windows(d1, d2, start, end, h_1):
    """
    Performs training for density estimation and graphs the Parzen windows
    :param d1: np.array (n x 2) - our training data for class 1
    :param d2: np.array (n x 2) - our training data class 2
    :param start:           int - min range of our training samples for creating window
    :param end:             int - max range of our training samples for creating window
    :param h_1:           float - window width
    :return: z1, z2: np.array (n x n) - likelihoods for both classes
    """
    # Start & end index * 10 to handle an iterable as a floating point
    w_start = start * 10
    w_end = (end * 10) + 1

    # Window width is 1 over the sqrt of training samples
    window_width = h_1 / np.sqrt(len(d1))  # 0.2
    g = [p / 10 for p in range(w_start, w_end)]
    p1, p2 = compute_parzen_window(d1, d2, g, window_width)

    x, y = np.meshgrid(g, g)
    z1 = np.reshape(p1, x.shape)
    z2 = np.reshape(p2, x.shape)
    plot_all(x, y, z1, z2)
    return z1, z2


def classify(x, z1, z2, g):
    """
    Classify a training sample using likelihood from Parzen Windows
    :param x:  np.array (1 x d) - Testing sample for classification
    :param z1: np.array (n x n) - Parzen window likelihood for Class 1
    :param z2: np.array (n x n) - Parzen window likelihood for Class 2
    :param g:      list [float] - set of window steps [-4.0, -3.9 ... , 3.9, 4.0]
    """
    # Get Index for each feature on step list
    x_1 = g.index(x[0])
    x_2 = g.index(x[1])
    # Get Likelihood for each class
    lh_1 = z1[x_1][x_2]
    lh_2 = z2[x_1][x_2]
    print('Likelihood for Class 1: {0:0.4e}'.format(lh_1))
    print('Likelihood for Class 2: {0:0.4e}'.format(lh_2))
    if lh_1 > lh_2:
        print('Sample [{}, {}] belongs to class 1'.format(x[0], x[1]))
    else:
        print('Sample [{}, {}] belongs to class 2'.format(x[0], x[1]))


# User-Defined Parameters
d1, d2 = get_data()
start, stop, h_1 = -4, 8, 2
z1, z2, = train_parzen_windows(d1, d2, start, stop, h_1)

# Re-create g for classification
g = [p / 10 for p in range(-40, 81)]
classify(np.array([1, -2]), z1, z2, g)
classify(np.array([7, 3]), z1, z2, g)