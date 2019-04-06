import pandas as pd


#r = pd.read_csv('hw3_1.csv')
#print(r.head())
#d = pd.read_csv('C:\\Users\\Justin\\PycharmProjects\\machine_learning\\pattern_classification\\hw3_1.csv', header=None)
#d = d.transpose()



#https://www.python-course.eu/expectation_maximization_and_gaussian_mixture_models.php
import numpy as np
from scipy.stats import norm

np.random.seed(0)
X = np.linspace(-5, 5, num=20)
X0 = X * np.random.rand(len(X)) + 10  # Create data cluster 1
X1 = X * np.random.rand(len(X)) - 10  # Create data cluster 2
X2 = X * np.random.rand(len(X))  # Create data cluster 3
X_tot = np.stack((X0, X1, X2)).flatten()  # Combine the clusters to get the random datapoints from above
"""Create the array r with dimensionality nxK"""
r = np.zeros((len(X_tot), 3))
print('Dimensionality', '=', np.shape(r))
"""Instantiate the random gaussians"""
gauss_1 = norm(loc=-5, scale=5)
gauss_2 = norm(loc=8, scale=3)
gauss_3 = norm(loc=1.5, scale=1)
"""
Probability for each datapoint x_i to belong to gaussian g 
"""
for c, g in zip(range(3), [gauss_1, gauss_2, gauss_3]):
    r[:, c] = g.pdf(X_tot)  # Write the probability that x belongs to gaussian c in column c.
    # Therewith we get a 60x3 array filled with the probability that each x_i belongs to one of the gaussians
"""
Normalize the probabilities such that each row of r sums to 1
"""
for i in range(len(r)):
    r[i] = r[i] / np.sum(r, axis=1)[i]
"""In the last calculation we normalized the probabilites r_ic. So each row i in r gives us the probability for x_i 
to belong to one gaussian (one column per gaussian). Since we want to know the probability that x_i belongs 
to gaussian g, we have to do smth. like a simple calculation of percentage where we want to know how likely it is in % that
x_i belongs to gaussian g. To realize this, we must dive the probability of each r_ic by the total probability r_i (this is done by 
summing up each row in r and divide each value r_ic by sum(np.sum(r,axis=1)[r_i] )). To get this,
look at the above plot and pick an arbitrary datapoint. Pick one gaussian and imagine the probability that this datapoint
belongs to this gaussian. This value will normally be small since the point is relatively far away right? So what is
the percentage that this point belongs to the chosen gaussian? --> Correct, the probability that this datapoint belongs to this 
gaussian divided by the sum of the probabilites for this datapoint for all three gaussians."""

print(r)
print(np.sum(r, axis=1))  # As we can see, as result each row sums up to one, just as we want it.

import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')
import numpy as np
from scipy.stats import norm

np.random.seed(0)
X = np.linspace(-5, 5, num=20)
X0 = X * np.random.rand(len(X)) + 10  # Create data cluster 1
X1 = X * np.random.rand(len(X)) - 10  # Create data cluster 2
X2 = X * np.random.rand(len(X))  # Create data cluster 3
X_tot = np.stack((X0, X1, X2)).flatten()  # Combine the clusters to get the random datapoints from above
"""Create the array r with dimensionality nxK"""
r = np.zeros((len(X_tot), 3))
print('Dimensionality', '=', np.shape(r))
"""Instantiate the random gaussians"""
gauss_1 = norm(loc=-5, scale=5)
gauss_2 = norm(loc=8, scale=3)
gauss_3 = norm(loc=1.5, scale=1)
"""Instantiate the random pi_c"""
pi = np.array([1 / 3, 1 / 3, 1 / 3])  # We expect to have three clusters
"""
Probability for each datapoint x_i to belong to gaussian g 
"""
for c, g, p in zip(range(3), [gauss_1, gauss_2, gauss_3], pi):
    r[:, c] = p * g.pdf(X_tot)  # Write the probability that x belongs to gaussian c in column c.
    # Therewith we get a 60x3 array filled with the probability that each x_i belongs to one of the gaussians
"""
Normalize the probabilities such that each row of r sums to 1 and weight it by pi_c == the fraction of points belonging to 
cluster c
"""
for i in range(len(r)):
    r[i] = r[i] / (np.sum(pi) * np.sum(r, axis=1)[i])
"""In the last calculation we normalized the probabilites r_ic. So each row i in r gives us the probability for x_i 
to belong to one gaussian (one column per gaussian). Since we want to know the probability that x_i belongs 
to gaussian g, we have to do smth. like a simple calculation of percentage where we want to know how likely it is in % that
x_i belongs to gaussian g. To realize this we must dive the probability of each r_ic by the total probability r_i (this is done by 
summing up each row in r and divide each value r_ic by sum(np.sum(r,axis=1)[r_i] )). To get this,
look at the above plot and pick an arbitrary datapoint. Pick one gaussian and imagine the probability that this datapoint
belongs to this gaussian. This value will normally be small since the point is relatively far away right? So what is
the percentage that this point belongs to the chosen gaussian? --> Correct, the probability that this datapoint belongs to this 
gaussian divided by the sum of the probabilites for this datapoint and all three gaussians. Since we don't know how many
point belong to each cluster c and threwith to each gaussian c, we have to make assumptions and in this case simply said that we 
assume that the points are equally distributed over the three clusters."""

print(r)
print(np.sum(r, axis=1))  # As we can see, as result each row sums up to one, just as we want it.

"""Plot the data"""
fig = plt.figure(figsize=(10, 10))
ax0 = fig.add_subplot(111)
for i in range(len(r)):
    ax0.scatter(X_tot[i], 0, c=np.array([r[i][0], r[i][1], r[i][2]]),
                s=100)  # We have defined the first column as red, the second as
    # green and the third as blue
for g, c in zip(
        [gauss_1.pdf(np.linspace(-15, 15)), gauss_2.pdf(np.linspace(-15, 15)), gauss_3.pdf(np.linspace(-15, 15))],
        ['r', 'g', 'b']):
    ax0.plot(np.linspace(-15, 15), g, c=c, zorder=0)

plt.show()