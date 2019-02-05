# https://machinelearningmastery.com/time-series-data-stationary-python/

from time_series import asset_returns as ar
from matplotlib import pyplot
df = ar.cleaning(ar.path)
df = ar.single_period_returns(df)
df = df.dropna()
df = df.set_index('date')

"""
Checks for stationarity include
1. Looking at the Plot
2. Summary Statistics
3. Statistical Tests

"""

# 1. Plotting
df['simple_gross_return'].plot()
pyplot.show()

"""
2. Summary statistics
   split time series into two or more partitions and compare the mean and variance of each group
"""

# By looking at mean and variance, we assume data confirms to Gaussian. Check with Histogram
df['simple_gross_return'].hist()
pyplot.show()

x = df['simple_gross_return'].values
split = len(x) // 2
x1, x2 = x[0:split], x[split:]
mean1, mean2 = x1.mean(), x2.mean()
var1, var2 = x1.var(), x2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))

"""
3. Augmented Dickey-Fuller Test
    -type of unit-root test
    - autoregressive model and optimizes an information criterion across multiple different lag values
    - null hypothesis - is that the time series can be represented by a unit root, that it is not stationary
        HO: If failed to be rejected, it suggests the time series has a unit root, meanining its non-stationary. It 
            has some time dependent structure
    - alternate hypothesis - the time series is stationary
        H1: The null hypothesis is rejected; it suggests the time series does not have a unit root, meaning its 
            stationary, and does not have a time-dependent structure
                      
    Interpret result using p-value from the test. A p-value below a threshold (5% or 1%) suggests we reject the null 
    hypothesis (stationary), otherwise a p-value above the threshold suggests we fail to reject the null hypothesis
    (non-stationary)
    
    p-value > 0.05 - Fail to reject HO
    p-value <= 0.05 - Reject HO
    
"""

from statsmodels.tsa.stattools import adfuller
X = df['simple_gross_return'].values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

"""
# ADF Statistic: -56.318393
# p-value: 0.000000
# Critical Values:
#  	1%: -3.432
#	5%: -2.862
#	10%: -2.567

Test Statistic = -56 
(The more negative, the more likely we are to reject the null hypothesis, we have stationary dataset)
                      
Our test statistics is less than the value of -3.432 at 1%

This suggests we can reject the null hypothesis with a significance level of less than 1%
Meaning a low probability that the result is a statistical fluke

When ADF test statistic is positive, it means we are much less likely to reject the null hypothesis

General
Positive ADF Statistic -> Reject Null -> Stationary
Negative ADF Statistic -> Do not Reject Null -> Non-Stationary
"""