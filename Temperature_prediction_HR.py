#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 00:40:36 2018

@author: klaudia
# PROBLEM LINK : https://www.hackerrank.com/challenges/temperature-predictions

Basically predict 'Missing_X vlaues'

"""
import numpy as np 


# DUMMY CASE: header:['yyyy', 'month', 'tmax' ,'tmin']
arr = np.array([
 ['1908', 'January' ,'5.0', '-1.4'],
 ['1908' ,'February' ,'7.3', 'Missing_22'],
 ['1945' ,'May' ,'15.4', '6.9'],
 ['1945' ,'June', '17.7', '9.8'],
 ['1945' ,'July', 'Missing_2', '12.1'],
 ['1945' ,'August', '23.7', '11.8'],
  ['1945' ,'September', '15.7', '6.8'],
 ['1945' ,'October', 'Missing_2', '4.1'],
 ['1945' ,'November', '7.7', '1.8']])


import numpy as np
from sklearn import ensemble

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False


m={"January":0,"February":1,"March":2,"April":3,"May":4,"June":5,"July":6,"August":7,"September":8,"October":9,
         "November":10,"December":11}

n =  arr.shape[0] #int(input()) # take 1st row
#input() # ignore 'header/ 2nd row 
## init storage vars for the feat of interest 
mins = []
maxs = []
x = [] # storage for year and date for complete data instances
testx = [] # storage for year and date of incomplete data instances (containing 'Missiing_num')

for i in range(n):
    line =  arr[i]#input().split('\t')
    # out : ['1908', 'January', '5.0', '-1.4'] ... ['1908', 'February', '7.3', '1.9'].. etc 
    maxs.append(float(line[2]) if isfloat(line[2]) else None)
    mins.append(float(line[3]) if isfloat(line[3]) else None)
    if isfloat(line[2]) and isfloat(line[3]):
        x.append([int(line[0]), m[line[1]]]) # use dictionary to convert from string to categorical 
    else:
        testx.append([int(line[0]), m[line[1]]])
        
ave_temp = ([(x + y)/2 for x, y in zip(maxs, mins) if x is not None and y is not None])        

model = ensemble.GradientBoostingRegressor()
model.fit(x, ave_temp) # where x is the yr and month
a = list(model.predict(testx))

# reverse the average eq to find the nan val:
for i in range(n):
    if mins[i] == None:
        print(2 * a.pop(0) - maxs[i]) # pop(0) selects the first item of list and pops it off the list so on next iterration can grab the next item etc 
    if maxs[i] == None:
        print(2 * a.pop(0) - mins[i])




####################################################################################
############ MY ORIGINAL TRIVIAL SOLUTION ##########################################
############ which didnt work... also because couldnt use fancyimpute package ######
####################################################################################
        
        
        
#tmax = [float(x) if x[0]!='M' else np.nan for x in arr[1:,2]]
#tmin= [float(x) if x[0]!='M' else np.nan for x in arr[1:,3]]
#
#
#df = pd.concat((pd.Series(tmax),pd.Series(tmin)),axis=1) 
#
#NaN_mask = df.isnull()
#
## from fancyimpute import KNN    # NOT SUPPORTED
## # X is the complete data matrix
## # X_incomplete has the same values as X except a subset have been replace with NaN
#
## # Use 3 nearest rows which have a feature to fill in each row's missing features
## df_filled_knn = KNN(k=3).complete(df)
#
##df2 = df_filled_knn[NaN_mask].dropna(axis=0, how='all')
#
## tried polynom fit of diff order but solution not good enough 
##df.interpolate(method='polynomial', order=1, inplace =True)
##df2 = df[NaN_mask].dropna(axis=0, how='all')
#
#r1 = df[NaN_mask][0].dropna()
#r2 = df[NaN_mask][1].dropna()
#
#idx1 = r1.index.values 
#idx2 = r2.index.values 
#
#len_r1 = len(r1)
#len_r2 = len(r2)
#
#for i in range(0,min(len_r1,len_r2)):
#    if idx1[i]> idx2[i]:
#        print(round(r2[idx2[i]],1))
#        print(round(r1[idx1[i]],1))
#    else:
#        print(round(r1[idx1[i]],1))
#        print(round(r2[idx2[i]],1))
#
#
#left = max(len_r1,len_r2) - min(len_r1,len_r2)
#
#if len_r1 > len_r2:
#    print(('{:.1f}\n'*left).format(*r1[-(len_r1-len_r2):]))
#else:
#    print(('{:.1f}\n'*left).format(*r2[-len_r2-len_r1:]))
#
#    
#        