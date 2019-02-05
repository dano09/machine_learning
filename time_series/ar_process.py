#https://campus.datacamp.com/courses/introduction-to-time-series-analysis-in-python/autoregressive-ar-models?ex=1
from time_series import asset_returns

from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_model import ARMA
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def simulate_ar_process(phi=0.9, plot=False):
    """
    # 0 lag coefficient of 1
    # sign of other coefficient is opposite from what we are using

    # Example, AR(1) process with phi = 0.9
    # the second element of AR array should be the opposite sign, - 0.9
    # Since ignoring MA at the moment, we just use 1

    :param phi:
    :return:
    """
    ar = np.array([1, -phi])
    ma = np.array([1])
    AR_object = ArmaProcess(ar, ma)
    simulated_data = AR_object.generate_sample(nsample=1000)
    if plot:
        plt.plot(simulated_data)
        plt.show()
    return simulated_data


def ar_example1():
    from statsmodels.tsa.arima_process import ArmaProcess

    # Plot 1: AR parameter = +0.9
    plt.subplot(2, 1, 1)
    ar1 = np.array([1, -0.9])
    ma1 = np.array([1])
    AR_object1 = ArmaProcess(ar1, ma1)
    simulated_data_1 = AR_object1.generate_sample(nsample=1000)
    plt.plot(simulated_data_1)

    # Plot 2: AR parameter = -0.9
    plt.subplot(2, 1, 2)
    ar2 = np.array([1, 0.9])
    ma2 = np.array([1])
    AR_object2 = ArmaProcess(ar2, ma2)
    simulated_data_2 = AR_object2.generate_sample(nsample=1000)
    plt.plot(simulated_data_2)
    plt.show()


def compare_ar_time_series(lags=20):
    sim_data_1 = simulate_ar_process()
    sim_data_2 = simulate_ar_process(-0.9)
    sim_data_3 = simulate_ar_process(0.3)
    plot_acf(sim_data_1, alpha=1, lags=lags)
    plot_acf(sim_data_2, alpha=1, lags=lags)
    plot_acf(sim_data_3, alpha=1, lags=lags)
    plt.show()


def estimate_ar_model(data, lookback=5400, forecast=5460,  ar_model_type=1, ma_model_type=0):
    """
    :param data - series with datetime index obj
    :param ar_model_type: 1 for AR(1)
    :param ma_model_type: MA model [not used here]
    :return:
    """

    mod = ARMA(data, order=(ar_model_type, ma_model_type))
    result = mod.fit()
    # To use date strings
    # index must be like numpy.datetime64()
    result.plot_predict(start=lookback, end=forecast)
    print(result.summary())
    plt.show()


def main():
    data = asset_returns.cleaning(asset_returns.path)
    df = asset_returns.single_period_returns(data)
    df = df.dropna()
    df = df.set_index('date')
    df.index = pd.to_datetime(df.index)
    df = df[::-1]
    estimate_ar_model(df.simple_gross_return)
