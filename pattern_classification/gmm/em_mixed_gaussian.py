import pandas as pd
from scipy.stats import multivariate_normal
import numpy as np
from functools import reduce
import matplotlib as mpl
import matplotlib.pyplot as pyplot

mpl.style.use('seaborn')
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def get_data():
    # data = pd.read_csv('C:\\Users\\Justin\\PycharmProjects\\machine_learning\\pattern_classification\\hw3_1.csv', header=None).transpose()
    data = pd.read_csv('hw3_1.csv', header=None).transpose()
    return data


def calc_cov_matrix(sample_vectors, mean_vector):
    """
    Params:
        sample_vector - list of Vectors ( n x 1 )
        mean_vector   - mean for each n ( n x 1)
    Returns
        scatter_matrix - Matrix ( n x n )
    """
    return reduce(lambda x, y: x+y, [np.outer(sample_vector - mean_vector, sample_vector - mean_vector) for sample_vector in sample_vectors]) / len(sample_vectors)


def calc_mean_vector(sample_vectors):
    """
    :param sample_vectors: list of np.array [ ( n x 1 ), ( n x 1 ), .., ( n x 1 ) ]
    :return: np.array - mean vector ( n x 1 )
    """
    # np.stack transforms the list into a multi-dimensional numpy array
    return np.stack(sample_vectors, axis=1).mean(axis=1)


def get_base_distributions(data):
    g1 = data[:50]
    g2 = data[50:]

    # MLE Parameters are the sample mean and sample covariance
    u1 = calc_mean_vector(g1.values)
    c1 = calc_cov_matrix(g1.values, u1)

    u2 = calc_mean_vector(g2.values)
    c2 = calc_cov_matrix(g2.values, u2)

    gauss1 = multivariate_normal(u1, c1)
    gauss2 = multivariate_normal(u2, c2)
    return gauss1, gauss2


def calc_expectation(data, gauss1, gauss2, pi):
    # E Step
    r = np.zeros((len(data), 2))
    for c, g, p in zip(range(2), [gauss1, gauss2], pi):
        r[:, c] = p * g.pdf(data)

    for i in range(len(r)):
        r[i] = r[i] / np.sum(r, axis=1)[i]

    # r is a matrix with a probability for each cluster (rows sum to 1)
    return r


def calc_total_weight(r):
    return r.sum(axis=0)


def calc_mixture(total_weight):
    # Parameter 1 (rho) Gaussian Mixture weighting
    return total_weight / total_weight.sum()


def calc_updated_mean(data, r, total_weight):
    """
    Calculate the updated mean

    Since our training sample has two features, we need to perform broadcast multiplication which
    will loop through each cluster vector in r and apply scalar multiplication to all samples for
    both features.
    :param data:     pd.DataFrame (n * 2) - training samples
    :param r:            np.array (n * 2) - our responsibility for each sample
    :param total_weight: np.array (1 * 2) - accumulation of responsibility for each cluster
    :return:     list of np.array (1 * 2) - mean vectors for each Gaussian
    """
    mu_c = []
    for cluster, rho in zip(r.T, total_weight):
        mu_c.append((np.sum(data.multiply(cluster, axis=0)) / rho).values)
    return mu_c


def calc_updated_cov(data, r, mu_u, total_weight):
    """
    Calculate the updated covariance

    For each cluster, we calculate the covariance matrix while also multiplying by each samples responsibility

    :param data:     pd.DataFrame (n * 2) - training samples
    :param r:            np.array (n * 2) - our responsibility for each sample
    :param mu_u: list of np.array (1 * 2) - mean vectors for each Gaussian
    :param total_weight: np.array (1 * 2) - accumulation of responsibility for each cluster
    :return:     list of np.array (2 * 2) - covariance matrix for each Gaussian
    """
    d = data.values
    cov_c = []
    for cluster, u, w in zip(r.T, mu_u, total_weight):
        s = 0
        for n in range(len(d)):
            s += cluster[n]*np.outer(d[n] - u, d[n] - u)
        cov_c.append(s / w)
    return cov_c


def calc_maximization(data, r):
    """
    Calculates the updated parameters based on our responsibility matrix
    :param data: pd.DataFrame (n * 2) - training samples
    :param r:        np.array (n * 2) - our responsibility for each sample
    :return: list of numpy.arrays
    """
    total_weight = calc_total_weight(r)
    rho_u = calc_mixture(total_weight)
    mu_u = calc_updated_mean(data, r, total_weight)
    cov_u = calc_updated_cov(data, r, mu_u, total_weight)
    return rho_u, mu_u, cov_u


def evaluate_log_likelihood(data, gauss1, gauss2, rho_u):
    """
    Computes the log-likelihood
    :param data: pd.DataFrame (n * 2) - training samples
    :param gauss1: scipy.stats._multivariate_normal
    :param gauss2: scipy.stats._multivariate_normal
    :param rho_u: list [float, float] - Gaussian Weights
    :return: float
    """
    r_new = np.zeros((len(data), 3))
    for c, g, p in zip(range(2), [gauss1, gauss2], rho_u):
        r_new[:, c] = p * g.pdf(data)

    r_new[:, 2] = np.log(r_new[:, 0] + r_new[:, 1])
    log_likelihood = r_new[:, 2].sum()
    return log_likelihood


def get_contour_grid_points(data):
    # Used for plotting the GMM
    d = data.values
    x, y = np.meshgrid(np.sort(d[:, 0]), np.sort(d[:, 1]))
    XY = np.array([x.flatten(), y.flatten()]).T
    return XY


def plot_gmm(data, XY, gauss1, gauss2, title):
    d = data.values
    fig = pyplot.figure(figsize=(10, 10))
    ax0 = fig.add_subplot(111)
    ax0.scatter(d[:, 0], d[:, 1])
    ax0.set_title(title)
    ax0.set_xlabel('Feature 1')
    ax0.set_ylabel('Feature 2')
    ax0.contour(np.sort(d[:, 0]),
                np.sort(d[:, 1]),
                gauss1.pdf(XY).reshape(len(d), len(d)),
                colors='black',
                alpha=0.3)
    ax0.contour(np.sort(d[:, 0]),
                np.sort(d[:, 1]),
                gauss2.pdf(XY).reshape(len(d), len(d)),
                colors='black',
                alpha=0.3)
    pyplot.show()


def em_for_mixed_gaussian(data, rho, epis, gmm_plot_data):
    """
    Compute paramters of GMM using the EM algorithm with log likelihood stopping criteria
    :param data: pd.DataFrame (n * 2)         - training samples
    :param rho: list [float, float]           - Gaussian Weights
    :param epis: float                        - threshold for stopping criteria
    :param gmm_plot_data: np.array ( n*n, 2 ) - for plotting the GMM
    """
    # Step 1, calculate initial parameters using subset of data
    gauss1, gauss2 = get_base_distributions(data)

    # Plot Starting Gaussians
    plot_gmm(data, gmm_plot_data, gauss1, gauss2, 'GMM - Initial State')

    # Log Likelihood
    l_hood = evaluate_log_likelihood(data, gauss1, gauss2, rho)

    # Difference in Log Likelihood from two iterations
    d_hood = np.inf

    iteration = 1
    print('Log-Likelihood')
    while d_hood > epis:

        # Step 2, perform Expectation step
        r = calc_expectation(data, gauss1, gauss2, rho)

        # Step 3, perform Maximization step
        rho, mu_u, cov_u = calc_maximization(data, r)

        # Step 4 Check for convergence
        gauss1 = multivariate_normal(mu_u[0], cov_u[0])
        gauss2 = multivariate_normal(mu_u[1], cov_u[1])

        u_hood = evaluate_log_likelihood(data, gauss1, gauss2, rho)
        d_hood = np.abs(l_hood - u_hood)
        print('{0:02d}: Original: {1:0.3f} Updated: {2:0.03f}, Difference: {3:0.03f}'.format(iteration, l_hood, u_hood, d_hood))
        l_hood = u_hood
        iteration += 1

    print('\nFinal Parameters:')
    print('rho: {}'.format(rho))
    print('M1 - Mean: {}'.format(mu_u[0]))
    print('M2 - Mean: {}'.format(mu_u[1]))
    print('M1 - Covariance: {}'.format(list(cov_u[0])))
    print('M2 - Covariance: {}'.format(list(cov_u[1])))

    # Plot Finished GMM
    plot_gmm(data, gmm_plot_data, gauss1, gauss2, 'GMM Finished State')


data = get_data()
rho = [0.5, 0.5]
epis = .001  # Stopping Threshold
XY = get_contour_grid_points(data)
em_for_mixed_gaussian(data, rho, epis, XY)