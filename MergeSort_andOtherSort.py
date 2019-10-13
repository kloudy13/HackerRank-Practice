#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 10:39:45 2018

SORTING ALGOS and Qs
"""


"""
Question: calc num of Min swaps to sort array
given an unordered array consisting of consecutive integers  [1, 2, 3, ..., n]
without any duplicates. You are allowed to swap any two elements. You need to find 
the minimum number of swaps required to sort the array in ascending order.
https://www.geeksforgeeks.org/minimum-number-swaps-required-sort-array/

"""
arr = [7, 1, 3, 2, 4, 5, 6]
ls = []

# Function returns the minimum  
# number of swaps required to sort the array 
def minSwaps(arr): 
    n = len(arr) 
# Create two arrays and use as pairs where first array is element and second 
# array is position of first element 
    arr_pos = [*enumerate(arr)] 
      
# Sort the array by array element values to get right position of  every element 
# as the elements of second array. 
    arr_pos.sort(key = lambda x: x[1]) 
      
# To keep track of visited elements. Initialize all elements as not visited or false. 
    visit = {k:False for k in range(n)} 
      
    # Initialize result 
    ans = 0
    for i in range(n): 
          
        # alreadt swapped or alreadt present at  correct position 
        if visit[i] or arr_pos[i][0] == i: 
            continue
              
        # find number of nodes  in this cycle and  add it to ans 
        cycle_size = 0
        j = i 
        while not visit[j]: 
              
            # mark node as visited 
            visit[j] = True
              
            # move to next node 
            j = arr_pos[j][0] 
            cycle_size += 1
              
        # update answer by adding 
        # current cycle 
        if cycle_size > 0: 
            ans += (cycle_size - 1) 
    # return answer 
    return ans r


########################################################################
############################ MergeSort  #################################
########################################################################
"""
MERGE SORT 

MergeSort(arr, l,  r) - recursive
If r > l
     1. Find the middle point to divide the array into two halves:  
             middle m = (l+r)/2
     2. Call mergeSort for first half:   
             Call mergeSort(arr, l, m)
     3. Call mergeSort for second half:
             Call mergeSort(arr, m+1, r)
     4. Merge the two halves sorted in step 2 and 3:
             Call merge(arr, l, m, r)
"""

# Python program for implementation of MergeSort 
def mergeSort(arr): 
    
    if len(arr) >1: 
#-------------------------- SORT PART ------------------------------------------
        
        mid = len(arr)//2 # find the middle of the array 
        L = arr[:mid] # split array elements into L/R halves 
        R = arr[mid:] # 
  
        mergeSort(L) # sort left half 
        mergeSort(R) # sort right half 
  
        i = j = k = 0
#-------------------------- MERGE PART -----------------------------------------
        
        while i < len(L) and j < len(R): 
            if L[i] < R[j]: 
                arr[k] = L[i] 
                i+=1
            else: 
                arr[k] = R[j] 
                j+=1
            k+=1
        # Checking if any element was left 
        while i < len(L): 
            arr[k] = L[i] 
            i+=1
            k+=1  
        
        while j < len(R): 
            arr[k] = R[j] 
            j+=1
            k+=1
  
# Code to print the list 
def printList(arr): 
    for i in range(len(arr)):        
        print(arr[i],end=" ") 
    print() 
  
# driver code to test the above code 
if __name__ == '__main__': 
    arr = [12, 11, 13, 9, 6, 7]  
    print ("Given array is", end="\n")  
    printList(arr) 
    mergeSort(arr) 
    print("Sorted array is: ", end="\n") 
    printList(arr) 
  
 # SAMPLE with OUTput to see whats going on  
 
 # Python program for implementation of MergeSort 
def mergeSort_print(arr): 
    
    if len(arr) >1: 
#-------------------------- SORT PART ------------------------------------------
        
        mid = len(arr)//2 # find the middle of the array 
        L = arr[:mid] # out [12, 11, 13]
        R = arr[mid:] # out [9, 6, 7]]
  
        mergeSort(L) # OUT  [11, 12, 13]
        mergeSort(R) # out [6, 7, 9]
  
        i = j = k = 0
#-------------------------- MERGE PART -----------------------------------------
        
        while i < len(L) and j < len(R): 
            if L[i] < R[j]: 
                arr[k] = L[i] 
                i+=1
                print('i:',i)
                print(arr)
            else: 
                arr[k] = R[j] 
                j+=1
                print('j:',j)
                print(arr)       
            k+=1
            print('k:',k)
            
#            j: 1
#            [6, 11, 13, 9, 6, 7]
#            k: 1
#            j: 2
#            [6, 7, 13, 9, 6, 7]
#            k: 2
#            j: 3
#            [6, 7, 9, 9, 6, 7]
#            k: 3   
#            and i = 0 so now the code below will go to the i < len(L) loop 
#               and fill the right half correct 
            
        # Checking if any element was left 
        while i < len(L): 
            arr[k] = L[i] 
            i+=1
            k+=1  
        
        while j < len(R): 
            arr[k] = R[j] 
            j+=1
            k+=1
  
# Code to print the list 
def printList(arr): 
    for i in range(len(arr)):        
        print(arr[i],end=" ") 
    print() 
  
# driver code to test the above code 
if __name__ == '__main__': 
    arr = [12, 11, 13, 9, 6, 7]  
    print ("Given array is", end="\n")  
    printList(arr) 
    mergeSort_print(arr) 
    print("Sorted array is: ", end="\n") 
    printList(arr)   
    
########################################################################
########################## Bubble Sort  #################################
########################################################################

    
# An optimized version of Bubble Sort 
def bubbleSort(arr): 
    n = len(arr) 
   
    # Traverse through all array elements 
    for i in range(n): 
        swapped = False
  
        # Last i elements are already 
        #  in place 
        for j in range(0, n-i-1): 
   
            # traverse the array from 0 to 
            # n-i-1. Swap if the element  
            # found is greater than the 
            # next element 
            if arr[j] > arr[j+1] : 
                arr[j], arr[j+1] = arr[j+1], arr[j] 
                swapped = True
  
        # IF no two elements were swapped 
        # by inner loop, then break 
        if swapped == False: 
            break
           
# Driver code to test above 
arr = [64, 34, 25, 12, 22, 11, 90] 
   
bubbleSort(arr) 

