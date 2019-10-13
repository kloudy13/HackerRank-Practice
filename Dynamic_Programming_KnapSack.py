#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 19:12:03 2018

@author: klaudia

Dynamic Porgramming 

(DP) is a technique that solves particular types of problems in Polynomial Time, 
ass oposed to the exponential brute method.

For a problem to be DP it needs to have:
1) Overlapping Subproblems
2) Optimal Substructure Property 
    
A given problems has Optimal Substructure Property if optimal solution of the 
given problem can be obtained by using optimal solutions of its subproblems.

Like Divide and Conquer, Dynamic Programming combines solutions to sub-problems. 
Dynamic Programming is mainly used when solutions of same subproblems are needed 
again and again. In dynamic programming, computed solutions to subproblems are 
stored in a table so that these don’t have to be recomputed. So Dynamic Programming 
is not useful when there are no common (overlapping) subproblems 

"""

"""
KNAPSACK PROBLEM
Given weights and values of n items, put these items in a knapsack of capacity W 
to get the maximum total value in the knapsack. In other words, given two integer 
arrays val[0..n-1] and wt[0..n-1] which represent values and weights associated with 
n items respectively. Also given an integer W which represents knapsack capacity, 
find out the maximum value subset of val[] such that sum of the weights of this 
subset is smaller than or equal to W.
"""

# A Dynamic Programming based Python Program for 0-1 Knapsack problem 
# Returns the maximum value that can be put in a knapsack of capacity W 
def knapSack(W, wt, val): 
    n = len(val)
    # init K as empty container:
    K = [[0 for x in range(W+1)] for x in range(n+1)]  # Out : 4 lists each of length 51 
  
    # Build table K[][] in bottom up manner 
    for i in range(n+1):  # 4
        #print (i)
        for w in range(W+1): # 50
            #print('w:',w)
            if i==0 or w==0: 
                K[i][w] = 0
                
            elif wt[i-1] <= w: # if weught is smaller than W
                K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]],  K[i-1][w]) 
                #print('K1:',K[i][w])
            else: 
                K[i][w] = K[i-1][w] 
                #print('K2:',K[i][w])
    return K[n][W] #last elem of K is ans
  
# Driver program to test above function 
val = [60, 100, 120] 
wt = [10, 20, 30] 
W = 50
#n = len(val) 
#print(knapSack(W, wt, val, n)) 
print(knapSack(W, wt, val)) 


"""
Dynamic programming Python implementation of Longest Increasiing Subsequence problem 
  
function lis returns length of the LIS in arr of size n 
 """
def lis(arr): 
    n = len(arr) 
  
    # Declare the list (array) for LIS and initialize LIS 
    # values for all indexes 
    lis = [1]*n 
  
    # Compute optimized LIS values in bottom up manner 
    for i in range (1 , n): 
        for j in range(0 , i): 
            if arr[i] > arr[j] and lis[i]< lis[j] + 1 : 
                lis[i] = lis[j]+1
  
    # Initialize maximum to 0 to get the maximum of all 
    # LIS 
    maximum = 0
  
    # Pick maximum of all LIS values 
    for i in range(n): 
        maximum = max(maximum , lis[i]) 
  
    return maximum 
# end of lis function 
  
# Driver program to test above function 
arr = [10, 22, 9, 33, 21, 50, 41, 60] 
print("Length of lis is", lis(arr) )

"""
Longest Common Subsequence (LCS)

Problem Statement: Given two sequences, find the length of longest subsequence 
present in both of them. A subsequence is a sequence that appears in the same 
relative order, but not necessarily contiguous. 

Example:
    1) LCS for input Sequences “ABCDGH” and “AEDFHR” is “ADH” of length 3.
    2) Consider the input strings “AGGTAB” and “GXTXAYB”. Last characters match 
    for the strings. So length of LCS can be written as:
        L(“AGGTAB”, “GXTXAYB”) = 1 + L(“AGGTA”, “GXTXAY”)
        
https://www.geeksforgeeks.org/wp-content/uploads/Longest-Common-Subsequence.png

"""

# Dynamic Programming implementation of LCS problem 
  
def lcs(X , Y): 
    # find the length of the strings 
    m = len(X) 
    n = len(Y) 
  
    # declaring the array for storing the dp values 
    L = [[None]*(n+1) for i in range(m+1)] 
  
    """Following steps build L[m+1][n+1] in bottom up fashion 
    Note: L[i][j] contains length of LCS of X[0..i-1] 
    and Y[0..j-1]"""
    for i in range(m+1): 
        for j in range(n+1): 
            if i == 0 or j == 0 : 
                L[i][j] = 0
            elif X[i-1] == Y[j-1]: 
                L[i][j] = L[i-1][j-1]+1
            else: 
                L[i][j] = max(L[i-1][j] , L[i][j-1]) 
  
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1] 
    return L[m][n] 
#end of function lcs 
  
  
# Driver program to test the above function 
X = "AGGTAB"
Y = "GXTXAYB"
print( "Length of LCS is ", lcs(X, Y) )



"""

More dynamic:
    - Game type: gold mine, moving on a matrix to collect max reward 
                https://www.geeksforgeeks.org/gold-mine-problem/
    - Permutation Coefficient: find value of P(n,k) = n!/(n-k)!
            https://www.geeksforgeeks.org/permutation-coefficient/
    - Bell Numbers (Number of ways to Partition a Set)!!
            https://www.geeksforgeeks.org/bell-numbers-number-of-ways-to-partition-a-set/
    - Friends Pairing Problem
        Given n friends, each one can remain single or can be paired up with some other friend.
        Each friend can be paired only once. Find out the total number of ways in which friends 
        can remain single or can be paired up.
            https://www.geeksforgeeks.org/friends-pairing-problem/
    
    

"""
