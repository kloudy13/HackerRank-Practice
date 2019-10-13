#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 18:56:35 2018

@author: klaudia
Given an n element array of integers: a , and an integer: m, 
determine the maximum value of the sum of any of its subarrays modulo m.
Ie. if m = 2, divide each sum(of set of a[i's])//2 (modulo two) and find the max of that/
Example:
  a = [3,5,9] m =7 
  [3] //7 = 3
  [5] //7 = 5
  [9] //7 = 2
  [3 5] sum to 8//7 = 1?
  [3 9] sum to 12//7 = 5
  [5 9] sum to 14//7 = 0
  [3 5 9] sum to 17//7 = 5
  
so max is 5

Key things to note: 
    -max value of max is m-1, so can stop algo when this is reached
 
    Solution: 
 https://www.geeksforgeeks.org/maximum-subarray-sum-modulo-m/
"""

m = 7
arr = [3, 3, 9, 9, 5]
a = [3,5,9] 


from bisect import insort, bisect_right

def maximumSum(a, m):
    # Create prefix tree
    prefix = [0] * len(a)
    curr = 0;
    for i in range(len(a)):
        curr = (a[i] % m + curr) % m
        prefix[i] = curr
    
    # Compute max modsum
    pq = [prefix[0]]
    maxmodsum = max(prefix)
    for i in range(1, len(a)):
        # Find cheapest prefix larger than prefix[i]
        left = bisect_right(pq, prefix[i])
        if left != len(pq):
            # Update maxmodsum if possible
            modsum = (prefix[i] - pq[left] + m) % m
            maxmodsum = max(maxmodsum, modsum)

        # add current prefix to heap
        insort(pq, prefix[i])

    return maxmodsum


import itertools as it

# below brute force encounters a problem... 
def maximumSum(arr, m):
    
    maximum = 0 
    set_arr = set(arr) 
    
    for element in list(set_arr):
        if element % m > maximum:
            maximum = element % m 
        if maximum == (m-1):
            break 
    for i in range(1,len(arr)):
        ls =[]  
        if i = 1:
            for a,b in it.combinations(arr, 2:
                ls.append((a,b,c))
                set_ls = set(ls)
                for element in list(set_ls):
                        if sum(element) % m > maximum:
                            maximum = sum(element)
                        if maximum == (m-1):
                            break    
        if i = 2:
            for a,b in it.combinations(arr, 2:
                ls.append((a,b,c))
                set_ls = set(ls)
                for element in list(set_ls):
                        if sum(element) % m > maximum:
                            maximum = sum(element)
                        if maximum == (m-1):
                            break   
        if i = 3:
            for a,b in it.combinations(arr, 2:
                ls.append((a,b,c))
                set_ls = set(ls)
                for element in list(set_ls):
                        if sum(element) % m > maximum:
                            maximum = sum(element)
                        if maximum == (m-1):
                            break    
    return maximum



