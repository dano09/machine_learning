import pandas as pd
import numpy as np




def readFile(filename):
    data = pd.read_csv(filename, sep=",", usecols=[0, 6], names=['Date', 'Price'], header=0)
    returns = np.array(data["Price"][:-1], np.float) / np.array(data["Price"][1:], np.float) - 1
    data["Returns"] = np.append(returns, np.nan)
    data.index = data["Date"]

    return data


att = "~/anaconda3/envs/interview_prep/machine_learning/CopyOfATT.csv"
coke = "~/anaconda3/envs/interview_prep/machine_learning/CopyOfKO2.csv"
spy = "~/anaconda3/envs/interview_prep/machine_learning/SPY.csv"

attData = readFile(att)
spyData = readFile(spy)
cokeData = readFile(coke)

# Done to return an array of arrays (each with 1 element
# Specific formatting for sklearn
xData = spyData["Returns"][0:-1].reshape(-1, 1)

print(xData)
#[[ 0.27192919]
# [-0.17106873]
# [-0.12372074]
# ...,
# [ 0.13934329]
# [-0.46921876]
# [ 0.70614791]]

yData = cokeData["Returns"][0:-1]
print(yData)

from sklearn import datasets, linear_model
goodCokeModel = linear_model.LinearRegression()
goodCokeModel.fit(xData, yData)

# Measure R-square by using score
rSquare = goodCokeModel.score(xData, yData)
print(rSquare)

# Determine coefficients
print(goodCokeModel.coef_)

# Determine intercept
print(goodCokeModel.intercept_)

# Determine residues
print(goodCokeModel.residues_)

import matplotlib.pyplot as plt
plt.scatter(xData, yData, color='black')
plt.plot(xData, goodCokeModel.predict(xData), color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()