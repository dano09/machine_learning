# https://www.youtube.com/watch?v=8mAZYv5wIcE

import pandas as pd
import numpy as np


# OLS = ( (X^T * X)^-1 ) * X^T * Y
x = pd.DataFrame(data=[[2, -1], [1, 2], [1, 1]])
y = pd.DataFrame(data=[[2], [1], [4]])


def least_squares(x, y):
    '''
        Least Squares Algorithm
        OLS = ( (X^T * X)^-1 ) * X^T * Y

        Assert: x - (number of rows > number of columns)
        (See Econometrics Greene)
        
        x - DataFrame input / independent / explanatory variables
        y - DataFrame outout / depedendnt / explained variables

        Returns DataFrame of coefficients
    '''
    shape = x.shape
    if shape[0] <= shape[1]:
        raise('Cant do lease squares without full rank of matrix X')

    x_transpose = x.T
    x_product_df = pd.DataFrame(np.dot(x_transpose, x))
    x_inv = pd.DataFrame(np.linalg.pinv(x_product_df.values))
    temp = x_inv.dot(x_transpose)
    least_squares_coefficients = temp.dot(y)

    return least_squares_coefficients