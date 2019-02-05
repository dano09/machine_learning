import numpy as np


def forward(obs_seq):
    T = len(obs_seq)
    N = A.shape[0]
    alpha = np.zeros((T, N))
    alpha[0] = pi*B[:, obs_seq[0]]

    for t in range(1, T):
        alpha[t] = alpha[t-1].dot(A) * B[:, obs_seq[t]]

    return alpha


def likelihood(obs_seq):
    # returns log P(Y \ mid model)
    # using the forward part of the forward-backward algorithm
    return forward(obs_seq)[-1].sum()