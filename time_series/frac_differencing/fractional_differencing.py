# Snippet 5.1 (pg 79)
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as pyplot
#import matplotlib as mpl
mpl.style.use('seaborn')

# Snippet 5.1 Weighting function (pg 79)
def get_weights(d, size):
    # threshold > 0 drops insignificant weights
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k*(d-k+1)
        w.append(w_)

    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def plot_weights(dRange, nPlots, size):
    w = pd.DataFrame()
    for d in np.linspace(dRange[0], dRange[1], nPlots):
        w_ = get_weights(d, size=size)
        w_ = pd.DataFrame(w_, index=range(w_.shape[0])[::-1], columns=[d])
        w = w.join(w_, how='outer')

    ax = w.plot()
    ax.legend(loc='upper right')
    pyplot.show()


#plot_weights(dRange=[0, 1], nPlots=5, size=6)
#plot_weights(dRange=[1, 2], nPlots=5, size=6)


# Snippet 5.2 Standard FRACDIFF (Expanding Window) (pg 82)
def frac_diff(series, d, thres=.01):
    '''
    Increasing width window, with treatment of NaNs
    Note 1: For thres=1, nothing is skipped
    Note 2: d can be any positive fractional, not necessarily bounded [0,1]
    :param series: pd.DataFrame
    :param d:
    :param thres:
    :return:
    '''

    # 1) Compute weights for the longest series
    w = get_weights(d, series.shape[0])

    # 2) Determine initial calcs to be skipped based on weight-loss threshold
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_ > thres].shape[0]

    # 3) Apply weights to values

    df = {}
    for name in series.columns:
        seriesF = series[[name]].fillna(method='ffill').dropna()
        df_ = pd.Series()
        for iloc in range(skip, seriesF.shape[0]):
            loc = seriesF.index[iloc]
            if not np.isfinite(series.loc[loc, name]):
                continue # exclude NAs
            df_[loc] = np.dot(w[-(iloc + 1):, :].T, seriesF.loc[:loc])[0,0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


# amzn = pd.read_csv('C:\\Users\\Justin\\PycharmProjects\\machine_learning\\time_series\\AMZN.csv')
amzn = pd.read_csv('AMZN.csv')
amzn = amzn[['Date', 'Adj Close']]
amzn.set_index('Date', inplace=True)
sample = amzn.tail(20)
res = frac_diff(sample, d=1)




# Snippet 5.3 The New Fixed-Width window FRACDIFF Method
def get_weights_ffd(d, thres):
    w = [1.]
    k = 1
    while True:
        w_ = -w[-1] / k * (d-k+1)
        if (abs(w_)) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)


def frac_diff_ffd(series, d, thres=1e-5):
    # Constant width window (new solution)
    w = get_weights_ffd(d, thres)
    width = len(w) - 1
    df = {}
    for name in series.columns:
        seriesF = series[[name]].fillna(method='ffill').dropna()
        df_ = pd.Series()
        for iloc1 in range(width, seriesF.shape[0]):
            loc0 = seriesF.index[iloc1 - width]
            loc1 = seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue  # exclude NAs
            df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


