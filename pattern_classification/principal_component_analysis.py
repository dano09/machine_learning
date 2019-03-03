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
        mean_vector   - np.array ( n x 1)
    Returns
        scatter_matrix - Matrix ( n x n )
    """

    #scatter_matrix = None
    #for sample_vector in sample_vectors:
    #    scatter_matrix += np.outer(sample_vector - mean_vector, sample_vector - mean_vector)

    # Functional
    return reduce(lambda x, y: x+y, [np.outer(sample_vector - mean_vector, sample_vector - mean_vector) for sample_vector in sample_vectors])


def perform_eigen_decomposition(scatte_matrix):
    """
    PCA transform vector { e } is simply the eigenvectors of the scatter matrix
    e - direction of the transformation
    lambda (eigenvalues) is the magnitude of the transformation
    :param scatte_matrix:
    :return: eigenvalues : np.array
             eigenvectors : np.array (2D) - column[:,i] is eigenvector corresponding to eigenvalue w[i]
    """
    #eigenvalues, eigenvectors = la.eig(scatte_matrix)
    return la.eig(scatte_matrix)
    #pca1 = max(eigenvalues)
    #projection = sample_vectors * pca1  # need to verify


def pca(training_data):
    """
    Steps for Principal Component Analysis
    1) Compute Mean Vector
    2) Calculate Scatter Matrix
    3) Perform eigen decomposition
    4) use max eigenvalue(s) to obtain the eigenvector e
    5) Solve for PCA ak = e * (x - m)

    params: training_data - list[np.array()]
    :return:
    """
    # mean vector
    m = calc_mean_vector(training_data)

    # scatter matrix
    s = calc_scatter_matrix(training_data, m)

    eigenvalues, eigenvectors = perform_eigen_decomposition(s)
    index_of_max = np.argmax(eigenvalues)
    e = eigenvectors[index_of_max]

    # Feature dimension
    d = len(training_data[0])

    #  Formula: a_k = e^(t) * (x - m)
    a = {k: np.dot(e.transpose(), (training_data - m)[k]) for k in range(0, d)}

    x = {k: m + a[k]*e for k in range(0, d)}
    print(a)

def hw2():
    d1 = [np.array([1, 2]), np.array([-3, -1]), np.array([4, 5]), np.array([-1, 1])]
    d2 = [np.array([0, -2]), np.array([5, 2]), np.array([-1, -4]), np.array([3, 1])]
    d = d1 + d2
    pca(d)

    #array([[26.75, 22.25],
    #       [22.25, 18.75]])

    #np.dot(sm, e)
    #array([[34.79391539, -0.09193335],
    #       [29.09661345, 0.10993448]])

hw2()