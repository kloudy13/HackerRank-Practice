#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 18:41:23 2018

make fun to return the number of distinct triplets that can be formed from 
the 3 given arrays. were p comes from arr a, q comes from arr b and 
r comes from arr c nd:
    p <= q and q >= r

"""


a = [1, 4, 5]
b = [2, 3, 3]
c = [1, 2, 3]



## Time OUT
#num = 0
#
#for elem in list(set(b)):
#    sub_a = [i for i in a if i <= elem]
#    sub_c = [i for i in c if i <= elem]
#    num += len(sub_a)*len(sub_c)
#
    
from bisect import bisect
def triplets(a, b, c):
    a, b, c = sorted(set(a)), sorted(set(b)), sorted(set(c))
    
    return sum([bisect(a, x) * bisect(c, x) for x in b])


def triplets2(a, b, c):
    a = list(sorted(set(a)))
    b = list(sorted(set(b)))
    c = list(sorted(set(c)))
    
    ai = 0
    bi = 0
    ci = 0
    
    ans = 0
    
    while bi < len(b):
        while ai < len(a) and a[ai] <= b[bi]:
            ai += 1
        
        while ci < len(c) and c[ci] <= b[bi]:
            ci += 1
        
        ans += ai * ci
        bi += 1
    
    return ans