#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 12:38:52 2018

@author: klaudia

Given an array, Find the element before which all the elements are smaller than it, 
and after which all are greater. Call this BALANCED ELEMENT 

Return index of the element if there is such an element, otherwise return -1.

Example:
Input:   arr[] = {5, 1, 4, 3, 6, 8, 10, 7, 9};
Output:  Index of element is 4
All elements on left of arr[4] are smaller than it
and all elements on right are greater.

APPROACH:
    
1) Create two arrays leftMax[] and rightMin[].
2) Traverse input array from left to right and fill leftMax[] such that leftMax[i] contains 
    maximum element from 0 to i-1 in input array.
3) Traverse input array from right to left and fill rightMin[] such that rightMin[i] contains 
    minimum element from to n-1 to i+1 in input array.
4) Traverse input array. For every element arr[i], check if arr[i] is greater than leftMax[i] 
    and smaller than rightMin[i]. If yes, return i.
    
"""

arr= [3,2,5,7,8]

arr= [5, 1, 4, 3, 6, 8, 10, 7, 9]

def balanced(arr):

    leftMax = 0
    rightMin = arr
    
    for i in range(1,len(arr)-1):
        leftMax = max(leftMax, arr[i-1])
#        print('i:',i,'arr_i',arr[i])
#        print('leftM:',leftMax)
#        print ('rightMin:',min(rightMin[i+1:]))

        if  leftMax < arr[i] < min(rightMin[i+1:]):
            ans = i 
            break
        else:
            ans = -1
            
    return ans



