#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 11:04:25 2018

@author: klaudia

 Pearson sample correlation from scratch 
"""

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

tmpm =  sum([(m - mean_M)**2 for m in M])
tmpp =  sum([(p - mean_P)**2 for p in P])
tmpc =  sum([(c - mean_C)**2 for c in C])

Corr_M_P = sum([lsmi*lspi for lsmi,lspi in zip(lsm,lsp)]) / (math.sqrt(tmpm)*math.sqrt(tmpp))
Corr_P_C = sum([lsci*lspi for lsci,lspi in zip(lsc,lsp)]) / (math.sqrt(tmpc)*math.sqrt(tmpp))
Corr_M_C = sum([lsmi*lsci for lsmi,lsci in zip(lsm,lsc)]) / (math.sqrt(tmpm)*math.sqrt(tmpc))

'''''''''''''''
Another implemntation unrelated to exoampel above but cleaner 
input 
'''''''''''''''
import math

def mean(x):
    return sum(x)/len(x)

def covariance(x,y):
    calc = []
    for i in range(len(x)):
        xi = x[i] - mean(x)
        yi = y[i] - mean(y)
        calc.append(xi * yi)
    return sum(calc)/(len(x) - 1)

def stDev(x):
    variance = 0
    for i in x:
        variance += (i - mean(x) ** 2) / len(x)
    return math.sqrt(variance)

def Pearsons(x,y):
    cov = covariance(x,y)
    return cov / (stDev(x) * stDev(y))

# driver
a = [1,2,3,4,5] ; b = [5,4,3,2,1]
print('covariance:',covariance(a,b))
print('Pearsons Corr:',Pearsons(a,b))
