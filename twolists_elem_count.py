#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 14:55:59 2018

@author: klaudia
 For each element in 1st array count elements less than or equal to it in 2nd array 

"""
  
def bin_search(arr, x): 
    """
    Function returns the index of largest element <= x in sorted array (arr). 
    If no such element exits function returns -1.
    For duplicates function returns the last index of occurrence of the element
    """
      
    low = 0
    hi = len(arr) - 1
    
    while(low <= hi): 
        
        mid = int((low + hi) / 2)  # find middle idx

        if( x >=  arr[mid]): # if x on the right, change low idx and search right side
            low = mid + 1; 
        else:  # else search left side
            hi = mid - 1

    return hi 
      


def count_smallerElements(arr1, arr2): 
    """
    for each element in 1st array (arr1):
    function counts elements less than or equal to it in 2nd array (arr2)

    """
    # sort the 2nd array 
    arr2.sort() 
      
    # for each element in first array 
    for i in range(len(arr1)): 
        # last index of largest element  
        # smaller than or equal to x 
        index = bin_search(arr2, arr1[i]) 
        # required count for the element arr1[i] 
        print(index + 1) 
      
        
# driver program to test above function 
arr1 = [1, 2, 3, 7, 9] 
arr2 = [0, 1, 2, 1, 1, 4]  # sorted:[0, 1, 1, 1, 2, 4]

count_smallerElements(arr1, arr2) 



"""
 For each element in 1st array count elements greater than it in 2nd array 

"""

def count_greaterElements(arr1, arr2): 
    """
    for each element in 1st array (arr1):
    function counts elements greater than (or equal to) it in 2nd array (arr2)

    """
    # sort the 2nd array 
    arr2.sort() 
      
    # for each element in first array 
    for i in range(len(arr1)): 
        # last index of largest element  
        # smaller than or equal to x 
        index = bin_search(arr2, arr1[i]) 
        print('index:',index)
        count = len(arr2)-index
        # required count for the element arr1[i] 
        print(count) 

arr1 = [1, 2, 3, 7, 9] 
arr2 = [0, 1, 2, 1, 1, 4]  # sorted:[0, 1, 1, 1, 2, 4]

count_greaterElements(arr1, arr2) 


