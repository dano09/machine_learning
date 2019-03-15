import numpy as np
from functools import reduce

def calc_weight(p, q):
    """
    wi = ln( pi(1-qi) / qi(1-pi) )
    :return:
    """
    wi = [np.log( (p[i]*(1-q[i])) / (q[i]*(1-p[i])) ) for i in range(len(p))]
    return wi

def calc_bias(prior_w1, prior_w2, p, q):
    """
    wo

    :return:
    """

    wo = reduce(lambda x, y: x+y, [np.log( (1-p[i]) / (1-q[i]) ) for i in range(len(p))]) + np.log(prior_w1 / prior_w2)
    return wo

def calc_disc(x, prior_w1, prior__w2, p, q):
    """
    g(x) = sum [ wi*xi + wo ]
    :return:
    """
    wi = calc_weight(p, q)
    wo = calc_bias(prior_w1, prior__w2, p, q)

    gx = reduce(lambda x, y: x+y, [wi[i]*x[i] for i in range(len(wi))]) + wo
    return gx

def main():
    """
    Decide w1 if g(x) > 0, o.w. decide w2

    Example is a two class problem with three independent binary features

    :return:
    """
    # Number of features d
    FEATURE_LENGTH = 3
    x = [0] * FEATURE_LENGTH

    # P(w_1) and P(w_2)
    prior_w1 = prior_w2 = 0.5

    # Potential for these not to be same values, but size should be same as x
    p = [0.8] * FEATURE_LENGTH
    q = [0.5] * FEATURE_LENGTH

    gx = calc_disc(x, prior_w1, prior_w2, p, q)
    # Decision Rule
    print('W1') if gx > 0 else print('W2')


import numpy as np
from functools import reduce
prior_w1 = prior_w2 = 0.5
p = [0.8, 0.8, 0.8]
q = [0.5, 0.5, 0.5]
reduce(lambda x, y: x+y, [np.log( (1-p[i]) / (1-q[i]) ) for i in range(len(p))]) + np.log(prior_w1 / prior_w2)