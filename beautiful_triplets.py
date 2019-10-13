#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 15:05:39 2018

@author: klaudia

Beutiful triplets of the form: (a_i,a_j,a_k)
where i<j<k and d is the distance s.t.: a_j-a_i = a_k -a_j = d

Given an increasing sequenc of integers and the value of d, count the number of
beautiful triplets in the sequence.

https://www.hackerrank.com/challenges/beautiful-triplets/problem
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



def beautifulTriplets(d, arr):

    l = len(arr)
    count = 0
    for i in range(0,l):
        
        j = binarySearch(arr, i, l-1, arr[i]+d ) # middle elem
        
        k = binarySearch(arr, j, l-1, arr[j]+d ) # right most elem
        
        if k != -1:
            count +=1
            
    return count

# driver test

d = 3
arr = [1,6,7, 7, 8, 10, 12, 13, 14, 19]#[1, 2 ,4, 5 ,7, 8, 10]
#       0 1  2  3  4  5  6

l = len(arr)
for i in range(0,l):
    j = binarySearch(arr, i, l-1, arr[i]+d )
#    if j == -1:
#        break
    k = binarySearch(arr, j, l-1, arr[j]+d )
    print(i,j,k)
#    if k == -1:
#        break    

    


# ALLL BELOW WORK BUT GIVE TIME OUT BECUSE TOO INEFFICIENT 
## Below code finds the idx location of all b.triplets
#
#l = len(arr)-1 #6
#store =[]
#for i in range(0,l+1):
#    print(i)
#    for j in range(i+1,l+1):
#        if arr[l-i] - arr[l-j] == d:
#            print(j)
#            for k in range(j+1,l+1):
#                if arr[l-j] - arr[l-k] == d:
#                    store.append([l-i,l-j,l-k])
#
#
## fun returns number of beautifulTriplets in an aray 
#def beautifulTriplets(d, arr):
#    
#    l = len(arr)-1
#    count = 0 
#    for i in range(0,l+1):
#        print(i)
#        for j in range(i+1,l+1):
#            if arr[l-i] - arr[l-j] == d:
#                print(j)
#                for k in range(j+1,l+1):
#                    if arr[l-j] - arr[l-k] == d:
#                        count += 1
#    return count 
#
