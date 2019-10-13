#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 15:02:03 2018

@author: klaudia

Greedy Algos 
 - Activity selection problem
 - Min Spanning Tree 

"""
import numpy as np 

"""The following implementation assumes that the activities 
are already sorted according to their finish time"""
  
"""Prints a maximum set of activities that can be done by a 
single person, one at a time
# n --> Total number of activities 
# s[]--> An array that contains start time of all activities 
# f[] --> An array that contains finish time of all activities 
"""
  
def printMaxActivities(s , f ): 
    n = len(f) 
    print( "The following activities are selected")
  
    # The first activity is always selected 
    i = 0
    print(i) 
  
    # Consider rest of the activities 
    for j in range(n): 
  
        # If this activity has start time greater than 
        # or equal to the finish time of previously 
        # selected activity, then select it 
        if s[j] >= f[i]: 
            print(j)
            i = j 
  
# Driver program to test above function 
s = [1 , 3 , 0 , 5 , 8 , 5] 
f = [2 , 4 , 6 , 7 , 9 , 9] 
printMaxActivities(s , f) 
  
# WHAT IF NOT SORTED, say have input:

ss = [5, 1 , 3 , 5, 0 , 8 ] #(start )
ff = [7, 2 , 4 , 9, 6 , 9 ] #(finish )


def sort_lists(ss,ff):
    """
    function sots two lists accoridng to the values in the second list (finish)
    """
    def takeSecond(elem):
        return elem[1]
    
    zipped = list(zip(ss,ff))
    
    zipped.sort(key=takeSecond)

    return list(zip(*zipped))[0], list(zip(*zipped))[1]

########################################################################
###################### Min Spanning Tree #############################
########################################################################
''' 
Spanning tree on a undirectional graph is a sub-grah connecting all the vertices
with min num of edges. A Minimum Spanning Tree is for graphs where edges have 
wieghts associated with them,  and is the spanning tree with the minimum sum of edge wieghts 
In other words.. 
----------------------------------------------------------------------------------------------------
What is Minimum Spanning Tree?
Given a connected and undirected graph, a spanning tree of that graph is a subgraph that 
is a tree and connects all the vertices together. A single graph can have many different 
spanning trees. A minimum spanning tree (MST) or minimum weight spanning tree for a weighted, 
connected and undirected graph is a spanning tree with weight less than or equal to the weight 
of every other spanning tree. The weight of a spanning tree is the sum of weights given to each 
edge of the spanning tree.

How many edges does a minimum spanning tree has?
A minimum spanning tree has (V – 1) edges where V is the number of vertices in the given graph.

----------------------------------------------------------------------------------------------------
ALGO WILL DEPEND ON INPUT; as graph can me represented by adjacency matrix or list
Time Complexity of the above program is O(V^2). If the input graph is represented 
using adjacency list, then the time complexity of Prim’s algorithm can be reduced
 to O(E log V) with the help of binary heap 

----------------------------------------------------------------------------------------------------
 Below is the Prim MST algo, which is v similar to Djiakstras shortest path algo 
 
 Dijkstra's algorithm doesn't create a MST (and can work on directed graphs),
 it finds the shortest path.

Consider this graph

       5     5
  S *-----*-----* T
     \         /
       -------
         9
The shortest path from S to T is 9, while the MST has to include all the nodes 
so it is a different 'path' at 10.
 
'''

# A Python program for Prim's Minimum Spanning Tree (MST) algorithm. 
# The program is for adjacency matrix representation of the graph 
  
import sys # Library for INT_MAX 
  
class Graph(): 
  
    def __init__(self, vertices): 
        self.V = vertices 
        self.graph = [[0 for column in range(vertices)]  
                    for row in range(vertices)] 
  
    # A utility function to print the constructed MST stored in parent[] 
    def printMST(self, parent): 
        print ("Edge \tWeight")
        for i in range(1,self.V): 
            print (parent[i],"-",i,"\t",self.graph[i][ parent[i] ] )
  
    # A utility function to find the vertex with  
    # minimum distance value, from the set of vertices  
    # not yet included in shortest path tree 
    def minKey(self, key, mstSet): 
  
        # Initilaize min value 
        min = sys.maxsize
  
        for v in range(self.V): 
            if key[v] < min and mstSet[v] == False: 
                min = key[v] 
                min_index = v 
  
        return min_index 
  
    # Function to construct and print MST for a graph  
    # represented using adjacency matrix representation 
    def primMST(self): 
  
        #Key values used to pick minimum weight edge in cut 
        key = [sys.maxsize] * self.V 
        parent = [None] * self.V # Array to store constructed MST 
        # Make key 0 so that this vertex is picked as first vertex 
        key[0] = 0 
        mstSet = [False] * self.V 
  
        parent[0] = -1 # First node is always the root of 
  
        for cout in range(self.V): 
  
            # Pick the minimum distance vertex from  
            # the set of vertices not yet processed.  
            # u is always equal to src in first iteration 
            u = self.minKey(key, mstSet) 
  
            # Put the minimum distance vertex in  
            # the shortest path tree 
            mstSet[u] = True
  
            # Update dist value of the adjacent vertices  
            # of the picked vertex only if the current  
            # distance is greater than new distance and 
            # the vertex in not in the shotest path tree 
            for v in range(self.V): 
                # graph[u][v] is non zero only for adjacent vertices of m 
                # mstSet[v] is false for vertices not yet included in MST 
                # Update the key only if graph[u][v] is smaller than key[v] 
                if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]: 
                        key[v] = self.graph[u][v] 
                        parent[v] = u 
  
        self.printMST(parent) 
  
g = Graph(5) 
g.graph = [ [0, 2, 0, 6, 0], 
            [2, 0, 3, 8, 5], 
            [0, 3, 0, 0, 7], 
            [6, 8, 0, 0, 9], 
            [0, 5, 7, 9, 0]] 
  
g.primMST(); 

########################################################################
###################### Dijkstra Shortest Path #############################
########################################################################
'''
https://startupnextdoor.com/dijkstras-algorithm-in-python-3/
https://www.bogotobogo.com/python/python_Dijkstras_Shortest_Path_Algorithm.php

'''
######################################################################
########## Greedy Algorithm to find Minimum number of Coins ##########
######################################################################
"""
    Find min number of coins/notes needed to give change of value V 
   1) Initialize result as empty.
   2) find the largest denomination that is smaller than V.
   3) Add found denomination to result. Subtract value of found denomination from V.
   4) If V becomes 0, then print result.  
        Else repeat steps 2 and 3 for new value of V
        
The Greedy solution wont work good for any arbitrary coin systems.
For example: if the coin denominations were 1, 3 and 4. To make 6, the greedy 
algorithm would choose three coins (4,1,1), whereas the optimal solution is two coins (3,3)
"""

def my_greedy_change(money, V):
    
    """
    fun to calcuate min number of coins/notes (stored in money list) needed to 
    give change of value V, where money should be a sorted list 
    """
    m = len(money)
    count = 0
    while V > 0:
        for i in range(m,0,-1):
            if money[i-1] <= V:
                mult = V // money[i-1] # calc how many times money fits in V 
                
                V = V -(mult * money[i-1])
                
                count += (1 * mult)
    return count 
               
# driver 
money = [1, 5, 10, 25]
V =111
       
my_greedy_change(money, V)
   
   

"""
  Dynamic programming approcah can be used in an 'arbitrary system' where the greedy method above fails 
"""
# Dynamic Programming Python implementation of MIN num coins to give change - RECURSIVE 

def minChangeDP(money, change, memory):
   min_coins = change # initialise 
   
   if change in money:
      memory[change] = 1  # if change denomination exists in money reurn 1 
      return 1
  
   elif memory[change] > 0:
      return memory[change]
  
   else:
       for i in [c for c in money if c <= change]:  # filter the list of coins to those less than the current value
         count = 1 + minChangeDP(money, change-i, memory)
         
         if count < min_coins:
            min_coins = count
            memory[change] = min_coins
            
   return min_coins

money = [1,5,10,25]
V = 111
memory = [0]*(V+1)

print(minChangeDP(money,V, memory))



# Dynamic Programming Python implementation of TOTAL num coins combinations to give change of V
def minCoinChange(Money, V):
    """
    function counts the min change using DP
    inputs: Money is the array of money, m is len(S), V is the value of the change.
    """
    m = len(Money)
    # list[i] will be storing the number of solutions for 
    # value i. We need V+1 length as the list is constructed 
    # in bottom up manner using the base case (n = 0) 
    # Initialize all values as 0 
    ls = [0]*(V+1)
  
    # Base case (If given value is 0) 
    ls[0] = 1
  
    # Pick all coins one by one and update the ls[] values 
    # after the index greater than or equal to the value of the 
    # picked coin 
    for i in range(0,m): # loop over len(money) 
        print('i:',i)
        for j in range(Money[i], V+1): 
            print('j:',j)
            ls[j] += ls[j- Money[i]] 
            print('ls:',ls)
  
    return ls[V] 
  
# Driver program to test above function 
money =  [1, 2, 5, 10, 20, 50, 100, 200]#[1, 2, 3] 
V = 100
x = minCoinChange(money, V) 
print (x) 
  

#####################################################################################
##### Greedy Algoto find Min/Max absolute difference between elements of array ##########
#####################################################################################
'''
   Naive approach woudl be to loop over list always keeping the min value (need 2 loops so On^2 compelxity
   
   Can be reduced to O(n Log n) if use sorting :

For Min:
1) Sort array in ascending order. This step takes O(n Log n) time.
2) Initialize difference as infinite. This step takes O(1) time.
3) Compare all adjacent pairs in sorted array and keep track of minimum difference. 
This step takes O(n) time.
   
'''
def my_min_abs_diff(arr):
    n = len(arr)
    # sort array/list 
    arr.sort()
    # init difference as very high
    min_abs_diff = 10**10
    # Compare all adjacent pairs in sorted array and keep track of minimum difference
    for i in range(1,n):
        abs_diff = abs(arr[i-1] - arr[i])
        if abs_diff< min_abs_diff:
            min_abs_diff = abs_diff
    return min_abs_diff
    

# Driver program to test above function
my_arr = [3, -7, 0]  
   
my_min_abs_diff(my_arr)



# My NAIVE implemntation from before
def my_max_abs_diff(a):
    
    diff = -10**10
    max_diff = -10**10
    n = len(a)
    a.sort()
    for i in range(0,n):
        for c in range(i+1,len(a) - i): 
            
            diff = abs(a[i] - a[i-c])
            
            if diff> max_diff:
                max_diff = diff
                
    return max_diff   

# Driver program to test above function
my_arr = [3, -7, 4,9,8,-1]  
   
my_max_abs_diff(my_arr)

#####################################################################################
##### Maximum difference between 2 elems of array s.t. smaller before larger ##########
#####################################################################################

# time out of course 
def my_diff(a,n):
    
    diff = -1
    max_diff = -1 # initialise to very low value
    
    for i in range(1,n):
        for j in range(0,i):
            if a[j] < a[i]:
                diff = a[i] - a[j]
                if diff> max_diff:
                    max_diff = diff
                
    return max_diff   

test = [2, 3, 10, 2, 4, 8, 1] #[3, 4, 10,1,8,3,  3, 10, 2, 4, 3,  3, 10, 2,1,8,7,6,5]#
n=len(test)
my_diff(test,n)

# Python 3 code to find Maximum difference 
# between two elements such that larger  
# element appears after the smaller number 
  
# The function assumes that there are  
# at least two elements in array. 
# The function returns a negative  
# value if the array is sorted in  
# decreasing order. Returns 0 if  
# elements are equal 
def maxDiff(a, n): 
    # initialise:
    min_val = a[0] 
    max_diff = a[1] - a[0] 
    
    for i in range(1,n): 
    # find the differnce between current elemnet and smallest element to the left of it:
        if max_diff < (a[i] - min_val): 
            max_diff = a[i] - min_val 
    # update min value if found new smallest value in array
        if min_val > a[i] :  
            min_val = a[i] 
      
    return max_diff 
      
# Driver program to test above function  

arr = [7,5,4,3]#[3, 4, 10,1,8,3, 1, 2, 6, 82, 105, 3, 10, 2, 4, 3,  3, 1, 2, 6, 80, 100,10, 2,1,8,7,6,5]
size = len(arr) 
print ("Maximum difference is",  
        maxDiff(arr, size)) 

def my_diff2(a,n):
    
    diff = -1
    max_diff = -1 # initialise to very low value

    m = max(a)
    idx = [i for i, j in enumerate(a) if j == m][0]
    for i in range(0,idx):
        diff = m - a[i]
        if diff> max_diff:
            max_diff = diff
                    
    for i in range(1,n):
        for j in range(0,i):
            if a[j] < a[i]:
                diff = a[i] - a[j]
                if diff> max_diff:
                    max_diff = diff
                
    return max_diff   


# My NAIVE implemntation from before
def my_max_abs_diff(a):
    diff = -10**10
    max_diff = -10**10
    
    for i in range(0,len(a)):
        for c in range(i+1,len(a) - i): 
            
            diff = abs(a[i] - a[c])
            
            if diff> max_diff:
                max_diff = diff
                
    return max_diff   
       
# My NAIVE implemntation from before
def lean_my_max_abs_diff(a):
    #diff = -10**10
    #max_diff = -10**10
    a.sort()
    max_diff = abs(a[-1] - a[0])
#    for i in range(1,len(a)):
#        diff = abs(a[i] - a[c])
#        
#        if diff> max_diff:
#            max_diff = diff
#                
    return max_diff   

# Driver program to test above function
other = [-59, -36, -13, 1 ,-53, -92, -2, -96, -54 ,75]
test = [0,5,90,-6]
#other_sorted =[-96, -92, -59, -54, -53, -36, -13, -2, 1, 75]
   
print(lean_my_max_abs_diff(other) == my_max_abs_diff(other))
                
print(lean_my_max_abs_diff(test) == my_max_abs_diff(test))
                       
     
       
### SIMPLE Max difference problem without absolute value part would only require
# to find the min and max elements of the list and substract samllest from largest.... 
    
"""       
Maximum absolute difference of value and index sums

Given an unsorted array A of N integers, A_{1}, A_{2}, ...., A_{N}. 
Return maximum value of f(i, j) for all 1 ≤ i, j ≤ N. Where: 
f(i, j) = |A[i] – A[j]| + |i – j|, where |A| denotes the absolute value of A.

Examples:

Calculate the value of f(i, j) for each pair of (i, j) and return the maximum value.
TO MAKE EFFICNET dont look over twice, istead make use of the property of abs value 
which means that ieL f(1, 2) = f(2, 1) and f(1, 1) = f(2, 2) = f(3, 3).

Input : A = {1, 3, -1}
Output : 5
f(1, 1) = f(2, 2) = f(3, 3) = 0
f(1, 2) = f(2, 1) = |1 - 3| + |1 - 2| = 3
f(1, 3) = f(3, 1) = |1 - (-1)| + |1 - 3| = 4
f(2, 3) = f(3, 2) = |3 - (-1)| + |2 - 3| = 5
So, we return 5.
   
EXPALINED: Cases 1 and 3 and cases 2 and 4 are the same!
Case 1: A[i] > A[j] and i > j
|A[i] - A[j]| = A[i] - A[j]
|i -j| = i - j
hence, f(i, j) = (A[i] + i) - (A[j] + j)

Case 2: A[i] < A[j] and i < j
|A[i] - A[j]| = -(A[i]) + A[j]
|i -j| = -(i) + j
hence, f(i, j) = -(A[i] + i) + (A[j] + j)

Case 3: A[i] > A[j] and i < j
|A[i] - A[j]| = A[i] - A[j]
|i -j| = -(i) + j
hence, f(i, j) = (A[i] - i) - (A[j] - j)

Case 4: A[i] < A[j] and i > j
|A[i] - A[j]| = -(A[i]) + A[j]
|i -j| = i - j
hence, f(i, j) = -(A[i] - i) + (A[j] - j)
"""   
   

  
# Function returns maximum absolue difference in linear time. 
def maxDistance(arr): 
      
    # max/ min variables to artificially low/high
    # in algorithm. 
    max1 = -10**10
    max2 = -10**10
    
    min1 = +10**10
    min2 = +10**10
   
    for i in range(len(arr)): 
        # Update max and min variables 
        max1 = max(max1, arr[i] + i) 
        min1 = min(min1, arr[i] + i) 
        max2 = max(max2, arr[i] - i) 
        min2 = min(min2, arr[i] - i) 
      
   
    # Calculating maximum absolute difference. 
    return max(max1 - min1, max2 - min2) 
  
   
# Driver program to test function above
  
arr = [ -70, -64, -6, -56, 64, 61, -57, 16, 48, -98 ] 
  
maxDistance(arr)
  

#####################################################################################
##### MAx ##########
#####################################################################################
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   