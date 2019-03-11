import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(1)


def ar3(a1, a2, a3, wnv, sample_size):
    """
    AR(3) model
    Params:
        a1  -       float : t-1 weight parameter
        a2  -       float : t-2 weight parameter
        a3  -       float : t-3 weight parameter
        wnv -         int : variance for our white noise
        sample_size - int : size of process
    Returns:
        numpy.Array - AR process
    """
    wn = np.random.normal(scale=np.sqrt(wnv), size=sample_size)  # Numpy takes SD
    r = np.zeros(sample_size)
    r[1], r[2] = 0.001, -0.001

    for t in range(3, sample_size):
        r[t] = 0.01 + a1 * (r[t - 1] - 0.01) + a2 * (r[t - 2] - 0.01) + a3 * (r[t - 3] - 0.01) + wn[t]

    return r

n_samples = 1000

# https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.normal.html
white_noise_variance = 0.0001
white_noise_sd = np.sqrt(white_noise_variance)  # Numpy takes SD
white_noise = np.random.normal(scale=white_noise_sd, size=n_samples)

alpha1 = 0.4
alpha2 = 0.3
alpha3 = 0.23
#ar_process = ar3(alpha1, alpha2, alpha3, white_noise, 2000)


from statsmodels.tsa.ar_model import AR
TEN_YEARS = 2500
data_10years = ar3(alpha1, alpha2, alpha3, white_noise_variance, TEN_YEARS)
model = AR(data_10years).fit(maxlag=3)
print(model)
print(type(model))
