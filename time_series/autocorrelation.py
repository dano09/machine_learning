# https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/

from pandas import Series
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from time_series import asset_returns as ar
import numpy as np


def plot_acf():
    df = ar.cleaning(ar.path)
    df = ar.single_period_returns(df)
    df = df.dropna()
    df = df.set_index('date')
    # Alpha - confidence interval
    # Example 0.05 means 5% chance that if true autocorrelation is zero, it will fall outside blue band
    plot_acf(df.simple_gross_return, alpha=0.5)
    pyplot.show()


    # Partial Correlation
    plot_pacf(df.simple_gross_return, lags=50)
    pyplot.show()


def generate_white_noise(plot=True):
    noise = np.random.normal(loc=0, scale=1, size=500)
    if plot:
        plot_acf(noise, lags=50)
        pyplot.show()
    return noise


def autocorrelation_function(ser, ts_lag=1):
    from statsmodels.tsa.stattools import acf
    """
    ACF - Calculate autocorrelation in a time-series

    :param ser: Pandas Series [column]
           ts_lag: int - number of days of lag for the ACF
    :return: Scalar Decimal of the autocorrelation
    """
    return ser.autocorr(lag=ts_lag)
    # return acf(ser)[ts_lag]


def test_autocorrelation(ser, ts_lag=365):
    """
    Ljung and Box Pierce Statistic to confirm autocorrelation

    Null Hypothesis: No Autocorrelation
    Alternative Hypothesis: Autocorrelation

    :param ser:
    :param ts_lag:
    :return:
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    results = acorr_ljungbox(ser, lags=ts_lag, boxpierce=True)
    test_statistics_ljung = results[0]
    p_values_ljung = results[1]
    test_statistics_box_pierce = results[2]
    p_value_bp = results[3]

    # if p_values_ljung is < .05, we can reject the null => autocorrelation
    # if p_values_ljung is < .05, we reject no autocorrelation => autocorrelation

    # if p_values_ljung is > .05, we cannot reject the null => maybe no autocorrelation
    # if p_values_ljung is > .05, we cannot reject no autocorrelation => maybe no autocorrelation
    return results


results = test_autocorrelation(df['simple_gross_return'])
test_statistics_ljung = results[0]
p_values_ljung = results[1]
test_statistics_box_pierce = results[2]
p_value_bp = results[3]