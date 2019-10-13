#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 16:59:20 2018

@author: klaudia

Given the words in the magazine and the words in the ransom note, print Yes if he 
can replicate his ransom note exactly using whole words from the magazine; otherwise,
print No.
For example, the note is "Attack at dawn". The magazine contains only "attack at dawn".
The magazine has all the right words, but there's a case mismatch. The answer is No.

"""

magazine = ['give', 'me', 'one', 'grand', 'today', 'today','night']

note = ['give', 'one','grand','today', 'today','today']


magazine = ['two', 'times', 'three', 'is', 'not', 'four']
note =  ['two', 'times', 'two', 'is', 'four']
note =  ['two', 'times', 'two', 'is', '5']

magazine = ['ive', 'got', 'a', 'lovely', 'bunch', 'of', 'coconuts']
note =['ive', 'got', 'some', 'coconuts']



def checkMagazine(magazine, note):
    
        l1 = len(note)
        l2 = len(magazine)
        
        magazine.sort()
        note.sort()
        
        i = 0
        j = 0
        count = 0
        
        while i<l1 and j<l2:
            
                if note[i] == magazine[j]:
                        count += 1
                        i += 1
                j += 1
                
        if count == l1:
                print ('Yes')
        else:
                print('No')





## TIME OUT
#def checkMagazine(magazine, note):
#    
#    if all(x in magazine for x in note) == True:
#        ans = 'Yes'
#        for elem in note:
#            if note.count(elem)>1:
#                if note.count(elem) > magazine.count(elem):
#                    ans = 'No'
#    else:
#        ans = 'No'
#     
#    return ans
#
#
