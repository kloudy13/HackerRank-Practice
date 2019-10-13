

"""
This is Polynom Regression problem from hacker rank
"""

from sklearn.linear_model import Ridge 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

import numpy as np
#import pandas as pd
#       [X1,    X2,   TARGET]
train = [[0.44, 0.68, 511.14], [0.99, 0.23, 717.1], [0.84, 0.29, 607.91], 
        [0.28, 0.45, 270.4], [0.07, 0.83, 289.88], [0.66, 0.8, 830.85], [0.73, 0.92, 1038.09],
        [0.57, 0.43, 455.19], [0.43, 0.89, 640.17], [0.27, 0.95, 511.06], [0.43, 0.06, 177.03], 
        [0.87, 0.91, 1242.52], [0.78, 0.69, 891.37], [0.9, 0.94, 1339.72], [0.41, 0.06, 169.88],
        [0.52, 0.17, 276.05], [0.47, 0.66, 517.43], [0.65, 0.43, 522.25], [0.85, 0.64, 932.21],
        [0.93, 0.44, 851.25], [0.41, 0.93, 640.11], [0.36, 0.43, 308.68], [0.78, 0.85, 1046.05], 
        [0.69, 0.07, 332.4], [0.04, 0.52, 171.85], [0.17, 0.15, 109.55], [0.68, 0.13, 361.97], 
        [0.84, 0.6, 872.21], [0.38, 0.4, 303.7], [0.12, 0.65, 256.38], [0.62, 0.17, 341.2], [0.79, 0.97, 1194.63], [0.82, 0.04, 408.6], [0.91, 0.53, 895.54], [0.35, 0.85, 518.25], [0.57, 0.69, 638.75], [0.52, 0.22, 301.9], [0.31, 0.15, 163.38], [0.6, 0.02, 240.77], [0.99, 0.91, 1449.05], [0.48, 0.76, 609.0], [0.3, 0.19, 174.59], [0.58, 0.62, 593.45], [0.65, 0.17, 355.96], [0.6, 0.69, 671.46], [0.95, 0.76, 1193.7], [0.47, 0.23, 278.88], [0.15, 0.96, 411.4], [0.01, 0.03, 42.08], [0.26, 0.23, 166.19], [0.01, 0.11, 58.62], [0.45, 0.87, 642.45], [0.09, 0.97, 368.14], [0.96, 0.25, 702.78], [0.63, 0.58, 615.74], [0.06, 0.42, 143.79], [0.1, 0.24, 109.0], [0.26, 0.62, 328.28], [0.41, 0.15, 205.16], [0.91, 0.95, 1360.49], [0.83, 0.64, 905.83], [0.44, 0.64, 487.33], [0.2, 0.4, 202.76], [0.43, 0.12, 202.01], [0.21, 0.22, 148.87], [0.88, 0.4, 745.3], [0.31, 0.87, 503.04], [0.99, 0.99, 1563.82], [0.23, 0.26, 165.21], [0.79, 0.12, 438.4], [0.02, 0.28, 98.47], [0.89, 0.48, 819.63], [0.02, 0.56, 174.44], [0.92, 0.03, 483.13], [0.72, 0.34, 534.24], [0.3, 0.99, 572.31], [0.86, 0.66, 957.61], [0.47, 0.65, 518.29], [0.79, 0.94, 1143.49], [0.82, 0.96, 1211.31], [0.9, 0.42, 784.74], [0.19, 0.62, 283.7], [0.7, 0.57, 684.38], [0.7, 0.61, 719.46], [0.69, 0.0, 292.23], [0.98, 0.3, 775.68], [0.3, 0.08, 130.77], [0.85, 0.49, 801.6], [0.73, 0.01, 323.55], [1.0, 0.23, 726.9], [0.42, 0.94, 661.12], [0.49, 0.98, 771.11], [0.89, 0.68, 1016.14], [0.22, 0.46, 237.69], [0.34, 0.5, 325.89], [0.99, 0.13, 636.22], [0.28, 0.46, 272.12], [0.87, 0.36, 696.65], [0.23, 0.87, 434.53], [0.77, 0.36, 593.86]]
#       [X1,    X2]
test = [[0.05, 0.54], [0.91, 0.91], [0.31, 0.76], [0.51, 0.31]]

train_arr = np.array(train)
test_arr = np.array(test)

X = train_arr[:, :-1]
y = train_arr[:, -1]

X_test = test_arr

# note X's look normalised already 
assert( 0<=min(train_arr[:,1]) and  max(train_arr[:,1])<=1)
assert( 0<=min(train_arr[:,0]) and  max(train_arr[:,0])<=1)

######################### ALternative  ##########################################

#import numpy as np
#import pandas as pd
#from sklearn import linear_model as lm
#from sklearn import preprocessing as pp
#
#F, N = map(int, input().split())
#train = np.array([input().split() for _ in range(N)], float)
#T = int(input())
#test = np.array([input().split() for _ in range(T)], float)
#
#mod = lm.LinearRegression()
#XtoP = pp.PolynomialFeatures(3, include_bias=False)
#mod.fit(XtoP.fit_transform(train[:, :-1]), train[:, -1])
#
#ymod = mod.predict(XtoP.fit_transform(test))
#print(*ymod, sep='\n')

######################### SIMPLE SOLUTION ##########################################

poly = PolynomialFeatures(degree = 3)
X_train_poly = poly.fit_transform(X)
X_test_poly = poly.fit_transform(X_test)

X_train, X_val, y_train, y_val = train_test_split(X_train_poly, y, random_state = 0)
# split into train and validation sets, use a fixed seed for reproducibility of 
# results, use default 75/25 split.
# The true test data will be held back until final model evaluation 

linreg = Ridge(alpha=0.5).fit(X_train, y_train)

#Addition of many polynomial features can lead to  overfitting, so use polynomial
#features in combination with regression that has a regularization penalty, like ridge
#regression. Default alpha is = 1

print('(poly + ridge) linear model coeff (w):\n{}'
     .format(linreg.coef_))
print('(poly + ridge) linear model intercept (b): {:.3f}'
     .format(linreg.intercept_))
print('(poly + ridge) R-squared score (training): {:.3f}'
     .format(linreg.score(X_train, y_train)))
print('(poly + ridge) R-squared score (validation): {:.3f}'
     .format(linreg.score(X_val, y_val)))

pred = linreg.predict(X_test_poly)

# print(('{:.2f}\n'*len(pred)).format(*pred)) desired HR out


######################### ADD CV  ##########################################

#    from sklearn.model_selection import GridSearchCV
#    from sklearn.linear_model import LogisticRegression
#    
#    grid_values = {'degree' : [2,3], 'aplha' : [0.25, 0.5, 0.75, 1] }
#                    # L1 or L2 regularization 
#    
#    lr = Ridge()
#    lr_gs = GridSearchCV(lr, param_grid = grid_values , scoring ='r2') #3 folds by default
#    
#    lr_gs.fit(X_train, y_train)
#    
#    res = lr_gs.cv_results_ # res is a giant dictionary of CV history
#    
#    return res['mean_test_score'] # pick relevant dict entry

true_out = [180.38,1312.07,440.13,343.72]


mse=((true_out-pred)**2).mean()

######################### Correlation ################################################
'''
implement Pearson sample correlation from scratch  (use fiest fromula )
https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
'''
sth = [[73.0, 72.0, 76.0], [48.0, 67.0, 76.0], [95.0, 92.0, 95.0], [95.0, 95.0, 96.0], [33.0, 59.0, 79.0], [47.0, 58.0, 74.0], [98.0, 95.0, 97.0], [91.0, 94.0, 97.0], [95.0, 84.0, 90.0], [93.0, 83.0, 90.0], [70.0, 70.0, 78.0], [85.0, 79.0, 91.0], [33.0, 67.0, 76.0], [47.0, 73.0, 90.0], [95.0, 87.0, 95.0], [84.0, 86.0, 95.0], [43.0, 63.0, 75.0], [95.0, 92.0, 100.0], [54.0, 80.0, 87.0], [72.0, 76.0, 90.0]]
N = 20

M = [row[0] for row in sth]
P = [row[1] for row in sth]
C = [row[2] for row in sth]

mean_M = sum(M)/len(M)
mean_P = sum(P)/len(P)
mean_C = sum(C)/len(C)

import math

lsm = [(m - mean_M) for m in M]
lsp = [(p - mean_P) for p in P]
lsc = [(c - mean_C) for c in C]

tmpm =  sum([(m - mean_M)**2 for m in M]) # sd = tmpm/(N-1)
tmpp =  sum([(p - mean_P)**2 for p in P])
tmpc =  sum([(c - mean_C)**2 for c in C])

Corr_M_P = sum([lsmi*lspi for lsmi,lspi in zip(lsm,lsp)]) / (math.sqrt(tmpm)*math.sqrt(tmpp))
Corr_P_C = sum([lsci*lspi for lsci,lspi in zip(lsc,lsp)]) / (math.sqrt(tmpc)*math.sqrt(tmpp))
Corr_M_C = sum([lsmi*lsci for lsmi,lsci in zip(lsm,lsc)]) / (math.sqrt(tmpm)*math.sqrt(tmpc))


"""
In probability theory and statistics, the coefficient of variation (CV), also known as 
relative standard deviation (RSD), is a standardized measure of dispersion of a probability 
distribution or frequency distribution. It is often expressed as a percentage, and is defined 
as the ratio of the standard deviation to the mean 
"""

CV_M = mean_M / (tmpm/(N-1)) 

