
"""
Spyder Editor

This is 1st problem form hacker rank: Correlation and Regression Lines - A Quick Recap #1
"""

import numpy as np 
import pandas as pd

from scipy.stats import pearsonr 
# returns: (Pearson’s correlation coefficient,2-tailed p-value)

x = [15, 12,  8,  8,  7,  7,  7,  6,  5,  3]
y = [10, 25, 17, 11, 13, 17, 20, 13, 9,  15]

per_corr, _ =  pearsonr(x,y)

print('{:.3f}'.format(per_corr))

######################################################################################3

"""
This is 2nd problem form hacker rank: Correlation and Regression Lines - A Quick Recap #2

Here are the test scores of 10 students in physics and history:
Physics Scores  15  12  8   8   7   7   7   6   5   3
History Scores  10  25  17  11  13  17  20  13  9   15
Compute the slope of the line of regression obtained while treating Physics as
 the independent variable. Compute the answer correct to three decimal places.
 """
 
from sklearn.linear_model import LinearRegression

 
linreg = LinearRegression().fit(np.array(x).reshape(-1, 1), np.array(y)) # train on whole dataset 
 
print('linear model coeff (w): {:.3f}'.format(linreg.coef_[0]))

######################################################################################3