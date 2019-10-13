#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 13:14:01 2018

@author: klaudia
# A sorting based program to  
# count pairs with difference k 
"""

# Standard binary search function  -> INPUT NEEDS TO BE A SORTED UNIQUE ARRAY 
# recursive implementation
def binarySearch(arr, low, high, x): 
    """
    Returns index of x in arr if present (between low and high index), else -1 
    """
    if (high >= low): 
     
        mid = low + (high - low)//2
        if x == arr[mid]: 
            return (mid) 
        elif(x > arr[mid]): 
            return binarySearch(arr, (mid + 1), high, x) 
        else: 
            return binarySearch(arr, low, (mid -1), x) 
      
    return -1
  
  
# Returns count of UNIQUE pairs with  
# difference k in arr[] of size n.  
def countPairsWithDiffK(arr, k): 
    n = len(arr)
# remove duplicates from arr[]   -- can skip if want to count ALL pairs ie not unique
    arr = list(set(arr))
    
    arr.sort() # Sort array elements 
  
    count = 0
    
    # Pick a first element point 
    for i in range (0, n - 1): 
        if (binarySearch(arr,  0,  n - 1, arr[i] + k) != -1): 
            count += 1
                  
    return count 

"""
Created on Thu Nov 15 13:14:01 2018

@author: klaudia
# A sorting based program to count pairs with SUM k 
"""

def countPairsWithSumK(arr, s): 
    n = len(arr)
  
# remove duplicates from arr[]   -- can skip if want to count ALL pairs ie not unique
    arr = list(set(arr))
    
    arr.sort() # Sort array elements 
  
    count = 0
    
    # Pick a first element point 
    for i in range (0, n-1 ):  #0 1 2 3 4
        if (binarySearch(arr,  0 ,  n-1 , (s - arr[i])) != -1): 
            count += 1
                  
    return count 


# Driver Code  
arr= [1, 3, 5, 8, 6, 4, 2]#[1, 5, 3, 4, 2] 
# 1, 2, 3, 4, 5, 6, 8
k = 2
print ("Count of pairs with given diff is ", 
             countPairsWithDiffK(arr, k))  
s= 6
print ("Count of pairs with given sum is ", 
             countPairsWithSumK(arr, s))  

# A simple program to count pairs with difference k 
  
def NAIVEcountPairsWithDiffK(arr, k): 
    count = 0
    n = len(arr)
    # Pick all elements one by one 
    for i in range(0, n): 
          
        # See if there is a pair of this picked element 
        for j in range(i+1, n) : 
              
            if arr[i] - arr[j] == k or arr[j] - arr[i] == k: 
                count += 1
                  
    return count 

# A simple program to count pairs with sum k  
def NAIVEcountPairsWithSumK(arr, k): 
    count = 0
    n = len(arr)
    
    # Pick all elements one by one 
    for i in range(0, n): 
          
        # See if there is a pair of this picked element 
        for j in range(i+1, n) : 
              
            if arr[i] + arr[j] == k: 
                count += 1
                  
    return count 
    










