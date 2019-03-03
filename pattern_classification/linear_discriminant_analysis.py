import numpy as np
from functools import reduce
import numpy.linalg as la


def calc_mean_vector(sample_vectors):
    """
    :param sample_vectors: list of np.array [ ( n x 1 ), ( n x 1 ), .., ( n x 1 ) ]
    :return: np.array - mean vector ( n x 1 )
    """
    # np.stack transforms the list into a multi-dimensional numpy array
    return np.stack(sample_vectors, axis=1).mean(axis=1)


def calc_scatter_matrix(sample_vectors, mean_vector):
    """
    Scatter Matrix is a Weighted Covariance Matrix
    AKA Pseudocovariance matrix
    Scatter = Covariance Matrix * ( n - 1 )

    Params:
        sample_vector - list of Vectors ( n x 1 )
        mean_vector   - mean for each n ( n x 1)
    Returns
        scatter_matrix - Matrix ( n x n )
    """
    return reduce(lambda x, y: x+y, [np.outer(sample_vector - mean_vector, sample_vector - mean_vector) for sample_vector in sample_vectors])


def calc_covariance_matrix(scatter, sample_size, bias=True):
    if bias:
        return scatter / sample_size
    else:
        return scatter / (sample_size - 1)


def calc_lda_projection(sw, sb):
    """
    LDA Projection is obtained as the solution of the generalized eigenvalue problem
    :return:
    """
    return la.eig(np.dot(la.inv(sw), sb))


def lda(d1, d2):
    # calculate mean by class
    u1 = calc_mean_vector(d1)
    u2 = calc_mean_vector(d2)

    s1 = calc_scatter_matrix(d1, u1)
    c1 = calc_covariance_matrix(s1, len(d1), bias=False)

    s2 = calc_scatter_matrix(d2, u2)
    c2 = calc_covariance_matrix(s2, len(d2), bias=False)

    # Within-class scatter matrix
    sw = c1 + c2

    # Between-class scatter matrix
    sb = np.outer(u1 - u2, u1 - u2)

    # Perform Projection
    eigenvalues, eigenvectors = calc_lda_projection(sw, sb)

    # Want max eigenvalue and corresponding projection
    index_of_max = np.argmax(eigenvalues)
    max_eigenvalue = eigenvalues[index_of_max]
    lda_projection = eigenvectors[:, index_of_max]
    print(max_eigenvalue)
    print(lda_projection)



def example():
    # http://www.sci.utah.edu/~shireen/pdfs/tutorials/Elhabian_LDA09.pdf
    d1 = [np.array([4, 2]), np.array([2, 4]), np.array([2, 3]), np.array([3, 6]), np.array([4, 4])]
    d2 = [np.array([9, 10]), np.array([6, 8]), np.array([9, 5]), np.array([8, 7]), np.array([10, 8])]
    lda(d1, d2)

def hw2():
    d1 = [np.array([1, 2]), np.array([-3, -1]), np.array([4, 5]), np.array([-1, 1])]
    d2 = [np.array([0, -2]), np.array([3, 2]), np.array([-1, -4]), np.array([3, 1])]


example()