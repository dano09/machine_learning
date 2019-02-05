# 2.6.3 General Multivariate normal case, covariance matrices are different for each category
import numpy as np
import numpy.linalg as la
from sympy.solvers import solve
from sympy.matrices import Matrix
from sympy import symbols
import matplotlib.pyplot as pyplot
import matplotlib as mpl

mpl.style.use('seaborn')


def calc_mean(x):
    return np.mean(x)


def calc_covariance(x, y, bias=True):
    """
    For samples, we want bias=False
    https://www.khanacademy.org/math/ap-statistics/summarizing-quantitative-data-ap/more-standard-deviation/v/review-and-intuition-why-we-divide-by-n-1-for-the-unbiased-sample-variance
    or use: np.cov  [unbiased by default]
    """
    if len(x) != len(y):
        return

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    sum = 0
    for i in range(0, len(x)):
        sum += ((x[i] - x_mean) * (y[i] - y_mean))
    if bias:
        return sum / len(x)
    return sum / (len(x) - 1)


def calc_weight_matrix(cov_mat):
    """ Wi in formula 67 pg.41 """
    return -0.5 * la.inv(cov_mat)


def calc_weight_vector(cov_mat, mean_vec):
    """ wi in formula 68 pg.41 """
    return la.inv(cov_mat).dot(mean_vec)


def calc_threshold(cov_mat, mean_vec, prior):
    """ wi0 in formula 69 pg.41 """
    t1 = -0.5 * (mean_vec.dot(la.inv(cov_mat))).dot(mean_vec)
    t2 = -0.5 * np.log(la.det(cov_mat))
    t3 = np.log(prior)
    return t1 + t2 + t3


def calc_discriminant(x, mean_vec, cov_mat, prior):
    """ tbd """
    Wi = calc_weight_matrix(cov_mat)
    wi = calc_weight_vector(cov_mat, mean_vec)
    wi0 = calc_threshold(cov_mat, mean_vec, prior)
    return x.transpose() * Wi * x + wi * x + wi0


def calc_decision_boundary(disc_func1, disc_func2, variable):
    return solve(disc_func1 - disc_func2, variable)


def calc_bhatt_bound(mean_vec1, mean_vec2, cov_mat1, cov_mat2):
    """
    Bhattacharyya Bound (pg 47) Eq - 77
    Assumes Gaussian dist.
    Provides upper bound on error
    :param mean_vec1, mean_vec2: numpy.array - mean vector
    :param cov_mat1, cov_mat1: numpy.array - covariance matrices
    :return: Constant value for k
    """
    avg_cov_matrix = (cov_mat1 + cov_mat2) / 2
    mean_diff = mean_vec2 - mean_vec1

    first_term = np.dot(np.dot(0.125*mean_diff, la.inv(avg_cov_matrix)), mean_diff)
    second_term = 0.5 * np.log(la.det(avg_cov_matrix) / np.sqrt(la.det(cov_mat1)*la.det(cov_mat2)))

    return first_term, second_term


def generate_2d_plot(result, variable, data1=None, data2=None, xst=-4, xls=12):
    x_vals = []
    y_vals = []
    for i in range(xst, xls):
        x_vals.append(i)
        y_vals.append(result.subs(variable, i))

    pyplot.plot(x_vals, y_vals)
    if data1 is not None and data2 is not None:
        pyplot.plot(data1[0], data1[1], 'o', label='C1')
        pyplot.plot(data2[0], data2[1], 'o', label='C2')
    pyplot.show(block=False)


def main(x, u1, cov1, u2, cov2, p1, p2, solve_for='x2'):
    h1 = calc_discriminant(x, u1, cov1, p1)
    h2 = calc_discriminant(x, u2, cov2, p2)
    result = calc_decision_boundary(h1, h2, solve_for)
    return result


def run_example():
    x1, x2 = symbols('x1 x2')
    x = Matrix([x1, x2])

    # Example page 40
    data1 = np.array([[2, 3, 3, 4], [6, 8, 4, 6]])
    data2 = np.array([[1, 3, 3, 5], [-2, 0, -4, -2]])

    u1 = np.array([calc_mean(data1[0]), calc_mean(data1[1])])
    cov1 = np.matrix([[calc_covariance(data1[0], data1[0]), calc_covariance(data1[0], data1[1])],
                      [calc_covariance(data1[1], data1[0]), calc_covariance(data1[1], data1[1])]])

    u2 = np.array([calc_mean(data2[0]), calc_mean(data2[1])])
    cov2 = np.matrix([[calc_covariance(data2[0], data2[0]), calc_covariance(data2[0], data2[1])],
                      [calc_covariance(data2[1], data2[0]), calc_covariance(data2[1], data2[1])]])

    # u1 = np.array([3, 6])
    # cov1 = np.matrix([[.5, 0], [0, 2]])
    # u2 = np.array([3, -2])
    # cov2 = np.matrix([[2, 0], [0, 2]])
    p1 = p2 = 0.5

    result = main(x, u1, cov1, u2, cov2, p1, p2, 'x2')
    generate_2d_plot(result[x2], x1, data1, data2)


def run_hw():
    # Homework
    x1, x2 = symbols('x1 x2')
    x = Matrix([x1, x2])
    u1 = np.array([0, 0])
    u2 = np.array([2, 2])
    cov1 = np.matrix([[1, 0], [0, 2]])
    cov2 = np.matrix([[1, 1], [1, 2]])
    p1 = 0.25
    p2 = 0.75
    result = main(x, u1, cov1, u2, cov2, p1, p2, 'x2')
    generate_2d_plot(result[0][x2], x1, xst=-9, xls=9)