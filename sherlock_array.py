#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:41:00 2018

@author: klaudia
Watson gives Sherlock an array of integers. His challenge is to find an element
of the array such that the sum of all elements to the left is equal to the sum
of all elements to the right.
You will be given arrays of integers and must determine whether there is an element that meets the criterion.
Function Description
Complete the balancedSums function in the editor below. It should return a string, either YES if there is an element meeting the criterion or NO otherwise.
balancedSums has the following parameter(s):
arr: an array of integers
"""
arr0 = [2 ,0 ,0, 0]#[1 ,2 ,4 ,1 ,1]

arr1 = [0, 2, 0, 2, 0, 0, 3, 1, 2, 4, 0, 0, 0, 1, 1, 2]
l = len (arr1)

arr2 = [0, 10, 0, 0, 0, 0, 0, 0, 2, 4, 0, 0, 0, 1, 1, 2]
l = len (arr2)

arr = [0, 1, 0, 0, 0, 0, 0, 0, 2, 4, 0, 0, 0, 1, 1, 2]
l = len (arr)

def balancedSums(arr):

    found = False
    left_sum = 0
    right_sum = sum(arr)
    
    for element in arr:
        right_sum -= element
        if(left_sum == right_sum):
            found = True
            break
        left_sum += element
        
    return print("YES" if found else "NO")



# Terminated due to timeout :( 
def naive_balancedSums(arr):
    l = len(arr)
    count = 0
    
    if l == 1:
        count = 1
    else:
        for i in range(0,l-1):
            left = sum(arr[:i])
            right = sum(arr[(i+1):])
            if left ==  right:
                count +=1
    if count == 0:
        ans = 'NO'
    else:
        ans = 'YES'
    return ans
    
            
            
            
        
        