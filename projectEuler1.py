#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Created on Sun Dec  2 19:56:17 2018

Find the sum of all the multiples of 3 or 5 below 1000.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#tot_sum = 0
#for num in range(1,1000):
#     if num%3 ==0 or num%5==0:
#         tot_sum += num

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
FIBONACCI
By considering the terms in the Fibonacci sequence whose values do not exceed 
four million, find the sum of the even-valued terms.

This may be a small improvement.  The Fibonacci series is:

1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610...

Now, replacing an odd number with O and an even with E, we get:

O, O, E, O, O, E, O, O, E, O, O, E, O, O, E...

And so each third number is even.  We don't need to calculate the odd numbers. 
 Starting from an two odd terms x, y, the series is:

x, y, x + y, x + 2y, 2x + 3y, 3x + 5y

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# moset elegant:
def even_fib_sum():
	x = y = 1
	sum = 0
	while (sum < 1000000):
		sum += (x + y)
		x, y = x + 2 * y, 2 * x + 3 * y
	return sum

# less elegant:
def even_fib_sum2():
    prev, cur = 0, 1
    total = 0
    while True:
        prev, cur = cur, prev + cur
        if cur >= 4000000:
            break
        if cur % 2 == 0:
            total += cur
    return total
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
What is the index of the first term in the Fibonacci sequence to contain 1000 digits?
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#############  LIST OF FIB NUMBERS IN A RANGE ###############
#  FIBo generator function
def fib():
    a,b = 0,1
    while True:
        yield a
        a, b = b, a + b

#and usage:
for index, fibonacci_number in zip(range(10), fib()):
     print('{i:3}: {f:3}'.format(i=index, f=fibonacci_number))

# or more useful: make a list of n fib numbers
n = 10 
f = fib()
[next(f) for i in range(0,n)]

#############  FIRST FIB NUMBER of X digits #################

def fist_Fib_ofLength(n):  # n is length of fib number (i.e number of digits)
    x, y = 1, 1
    ls = [x,y]
    while len(str(ls[-1]))< n :
        x, y = y, x+y
        ls.append(y)
    return len(ls) #ls[-1] returns 1st fib number with n digits (coudl return whole list)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
The prime factors of 13195 are 5, 7, 13 and 29.

What is the largest prime factor of the number 600851475143 ?
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def is_prime(n):
    '''
    function checks if number is prime 
    '''
    for i in range(3, n):
        if n % i == 0:
            return False
    return True

def find_prime_factors(num):
    '''
    function makes a list of prime factors <= num
    '''
    prime = []
    i = 2
    while True:
        if num <= prime[-1]:
            break
        if num % i == 0:
            num = num / i
            if is_prime(i) and i >= prime[-1]:
                prime.append(i)
        i += 1
    return prime

prime_list = find_prime_factors(600851475143)

print("biggest prime = %d" % prime_list[-1] )      


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Listing the first six prime numbers: 2, 3, 5, 7, 11, and 13, we can see that the 6th prime is 13.

What is the 10001st prime number?

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
## 10 times slower than below 
#def listPrime(n):
#    '''fun finds list of prime number below or equal n'''
#    ls = [2]
#    i = 3
#    while ls[-1] < n:
#        if is_prime(i):
#            ls.append(i)
#        i+=2
#    if ls[-1] > n:
#        ls.pop()
#    return ls

def makelistPrime(n):
    '''fun finds list of prime number below or equal n'''
    out = [2]
    for num in range(3, n+1, 2):
        if all(num % i != 0 for i in range(2, int(num**.5 ) + 1)):
            out.append(num)
    return out
'''
  sieve by repeatedly casting out multiples of primes. Begin by making a list of all 
  numbers from 2 to the maximum desired prime n. Then repeatedly take the smallest
  uncrossed number and cross out all of its multiples; the numbers that remain uncrossed are prime.
  For example, consider the numbers less than 30. Initially, 2 is identified as prime, 
  then 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28 and 30 are crossed out.
  Next 3 is identified as prime, then 6, 9, 12, 15, 18, 21, 24, 27 and 30 are crossed out. 
  The next prime is 5, so 10, 15, 20, 25 and 30 are crossed out. And so on. The numbers 
  that remain are prime: 2, 3, 5, 7, 11, 13, 17, 19, 23, and 29.
'''
def sievePrime(n):
    ls = []
    sieve = [True] * (n+1)
    for num in range(2, n+1):
        if (sieve[num]):
            ls.append(num)
            for i in range(num, n+1, num):
                sieve[i] = False
    return ls

# NAIVE 
def idxPrime(n):
    '''fun finds prime number with nth index where count starts at 1'''
    ls = [2]
    i = 3
    while len(ls) < n:
        if is_prime(i):
            ls.append(i)
        i+=2
    return ls[-1]
# --- 106.54631876945496 seconds ---

def nth_prime_number(n):
    if n==1:
        return 2
    count = 1
    num = 3
    while(count <= n):
        if is_prime(num):
            count +=1
            if count == n:
               return num
        num +=2 #optimization
        
# --- 105.111496925354 seconds ---

import time
start_time = time.time()
sievePrime(1110)
print("--- %s seconds ---" % (time.time() - start_time))   

''''''''''''''''''''' TIME IT!!'''''''''''''''''''''''''''
#import time
#start_time = time.time()
#main()
#print("--- %s seconds ---" % (time.time() - start_time))  


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Sum of the primes below N

eg. Sum of the primes below 10 is 2 + 3 + 5 + 7 = 17.

Find the sum of all the primes below two million.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
print('Sum of all the primes below two million.:', np.sum(sievePrime(2*10**6)))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PALINDROMES 

Largest palindrome product

A palindromic number reads the same both ways. 
The largest palindrome made from the product of two 2-digit numbers is 9009 = 91 × 99.

Find the largest palindrome made from the product of two 3-digit numbers.

note on halving: For example the number 69696 is checked both when a=132 and b=528 
and when a=528 and b=132. To stop checking numbers like this we can assume a ≤ b, roughly
halving the number of calculations needed (done in 2nd while loop in code)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def isPalindrome(n):
    return n == int(str(n)[::-1])

def largestPalindrome(n):
    '''
    fun finds largest palindrome which is a product of n digit numbers
    '''
    largestPalindrome = 0
    a = 10*n -1
    while a >= 100:
        b = 10*n -1
        while b >= a: # halve the num of configs becuse limit b to be > than a 
            if a*b <= largestPalindrome:
                break #Since a*b is always going to be too small
            if isPalindrome(a*b):
                largestPalindrome = a*b
            b = b-1
        a = a-1
    return largestPalindrome


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Double-base palindromes

The decimal number, 585 = 10010010012 (binary), is palindromic in both bases.

Find the sum of all numbers, less than one million, which are palindromic in base 10 and base 2.

(Please note that the palindromic number, in either base, may not include leading zeros.)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def dec_to_bin(x):
    return int(bin(x)[2:])

tot_sum = 0
for i in range(0,10**6):
    if ( isPalindrome(i) and isPalindrome(dec_to_bin(i))):
        #print(i)
        tot_sum += i
print('sum of numbers, less than 1 million, palindromic in base 10 and base 2:',tot_sum)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Smallest multiple (LCM - LOWEST COMMON MULTIPLE of a Set)

2520 is the smallest number that can be divided by each of the numbers from 
1 to 10 without any remainder. What is the smallest positive number that is 
evenly divisible by all of the numbers from 1 to 20?

Observe:
  -  needs to end in 0 (divisible by 20,15,10,5)
  - A number is divisible by 2  if the last digit is 0, 2, 4, 6 or 8. ie even 
  - A number is divisible by 3  if the sum of the digits is divisible by 3.
  - A number is divisible by 4  if the number formed by the last two digits is divisible by 4.
  - A number is divisible by 5  if the last digit is either 0 or 5.
  - A number is divisible by 6  if it is divisible by 2 AND it is divisible by 3.
  - A number is divisible by 8  if the number formed by the last three digits is divisible by 8.
  - A number is divisible by 9  if the sum of the digits is divisible by 9. 
  - divisible by 10 if ends in 0
  - divisible by 20 if It is divisible by 10, and the tens digit is even.
  and more ....

can generate numbers in increments of 20
  
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
divisors = [20,19,18,17,16,15,14,13,12,11,9,8,7,4]# can reduce #divsiors using knwonw rules above

num = 2520 # start from largest known which does not meet criteria 

while all( num%i == 0 for i in divisors) == False:
    num += 20
print('smallest num:',num) 

'''
gcd(x, y) -> int greatest common divisor of x and y is a inbuilt function 
'''
def gcd(a, b): # note math.gcd(a,b) exisits 
    """Return greatest common divisor using Euclid's Algorithm."""
    while b:      
        a, b = b, a % b
    return a

def lcm(a, b):
    """Return lowest common multiple."""
    return a * b // gcd(a, b)

from functools import reduce
def lcmm(*args):
    """Return lcm of args.
    
        You can compute the LCM of more than two numbers by iteratively computing
        the LCM of two numbers, i.e. lcm(a,b,c) = lcm(a,lcm(b,c))
    """   
    return reduce(lcm, args)

''' USING REDUCE FUN: apply a fun to each elem of list 
EXAMPLE: 
    Naive product of list elemsnts:
product = 1
list = [1, 2, 3, 4]
for num in list:
    product = product * num

# Output: product = 24
 
    OR 
product = reduce((lambda x, y: x * y), [1, 2, 3, 4])

# Output: product = 24
'''


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Sum square difference

Find the difference between the sum of the squares of the first one hundred natural 
numbers and the square of the sum.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np 
x = np.arange(1,101)
#x = np.arange(1,11)

def sum_suare(ls):
    return np.sum(ls)**2
def square_sum(ls):
    return np.sum([x**2 for x in ls])

diff = sum_suare(x) - square_sum(x)
print('difference is:',diff)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
A Pythagorean triplet is a set of three natural numbers, a < b < c, for which,

a^2 + b^2 = c^2

There exists exactly one Pythagorean triplet for which a + b + c = 1000.
Find the product abc.    
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# my code
import math 
start_time = time.time()
for a in range(1, 1000):
     for b in range(a, 1000):
         c = math.sqrt(a**2 + b**2)
         if c == (1000-a-b):
             print ('ans:',int(a*b*c))
print("--- %s seconds ---" % (time.time() - start_time))  

# faster code:
start_time = time.time()
for a in range(1, 1000):
     for b in range(a, 1000):
         c = 1000 - a - b
         if c > 0: # makes faster 
             if c*c == a*a + b*b: # only calc sometimes and not use sqrt()
                 print (a*b*c)
                 break
print("--- %s seconds ---" % (time.time() - start_time))  

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Highly divisible triangular number

The sequence of triangle numbers is generated by adding the natural numbers. 
So the 7th triangle number would be 1 + 2 + 3 + 4 + 5 + 6 + 7 = 28. The first ten terms would be:

terms : 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, ...
indx:  +1, +2 +3 +4  +5  +6  +7  +8
Let us list the factors of the first seven triangle numbers:

 1: 1
 3: 1,3
 6: 1,2,3,6
10: 1,2,5,10
15: 1,3,5,15
21: 1,3,7,21
28: 1,2,4,7,14,28
We can see that 28 is the first triangle number to have over five divisors.

What is the value of the first triangle number to have over five hundred divisors?
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def make_triangular_number(n):
    ''' fun makes list of n triangular numbers tsarting form 1'''
    ls = [1]
    for i in range(1,n):
        ls.append(ls[i-1]+i+1)
    return ls
    
import math
def divisorGenerator(n):
    ''' fun to get all divisors of a number, NP hard problem!, used 4encryption'''
    large_divisors = []
    for i in range(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i*i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield divisor

print(list(divisorGenerator(100)))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Coin sums

coin pool : 1p, 2p, 5p, 10p, 20p, 50p, £1 (100p) and £2 (200p).

How many different ways can £2 be made using any number of coins?
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

s = set(1, 2, 5, 10, 20, 50, 100, 200)


from itertools import combinations_with_replacement
coins = [1, 2, 5, 10, 20, 50, 100,200]

# Sadly O(n2) is too slow 
# ONE LINE
result = [seq for i in range(len(coins), 0, -1) for seq in combinations_with_replacement(coins, i) if sum(seq) == 200]
print ("Ways to make change =", len(result))
# same in more lines: 
ls = []
for i in range(1,len(coins)):
    for seq in combinations_with_replacement(coins, i):
        if sum(seq) == 200:
            ls.append(seq)
print ("Ways to make change =", len(ls))

# fast using dynamic programming 
# explained: https://www.xarg.org/puzzle/project-euler/problem-31/     
target = 100
coins = [1, 2, 5, 10, 20, 50, 100, 200]
ways = [1] + [0]*target # len 201

for coin in coins:
    print('coin:',coin)
    for i in range(coin, target+1):
        print('i:',i)
        ways[i] += ways[i-coin]
        print(ways)

print( "Ways to make change =", ways[target])


#target = 10
#coins = [1, 2, 5, 10]#, 20, 50, 100, 200]
#ways = [1] + [0]*len(coins) # len 201
#
#for coin in coins:
#    print('coin:',coin)
#    for i in range(1, len(coins)):
#        print('i:',i)
#        ways[i] += ways[i-1]
#        print(ways)
#
#print( "Ways to make change =", ways[target])
