#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 21:36:02 2021

@author: nathan
"""
####PART 4####


#MORE ON STRINGS
#min function: lets start with tuple
example = (3,4,5)
min(example)
#gives us min
#applied to tuple containing strings: we will get a because each symbol has unique "ASCII value"
characters = ('a','b','c')
min(characters)
#to check ASKII value, use ord function
ord('a')
ord('b')

#LIST COMPREHENSION (makes code cleaner and more efficient)
#consider for loop
for i in range(0,5):
    print(i)
#store elements in a list:
list = []
for i in range(0,5):
    list.append(i)
#Using list comprehension:
list1 = [i for i in range(0,5)]
list1
#as float:
list2 = [float(i) for i in range(0,5)]
list2

#if statements & list comprehension:
list3 = [i for i in range(0,5) if i > 2 and i < 5]  
list3  
list4 = [i*2 if i > 2 else i*3 for i in range(0,5)]
list4

#if statements and input function:
shoppinglist = [input('What item? ') for i in range(0,3)]
shoppinglist

#list of lists:
listoflist = [[i for i in range(0,5)] for i in range (0,5)]
listoflist

#NESTED LOOPS
#1st example: we want to print 0,1,2 three times 
for i in range(3):
    for j in range(3):
        print(i,j)
        #notice how this executes, 1 loop within the other.
#next task: create list of integers dates jan2010-dec2011
Dates = []
start = 201001
for i in range(2):
    for j in range(12):
        Dates.append(start+j)
    start = start + 100
Dates    

#using nested loops to create structures
#triangle:first understand end arguemnt
for i in range(3):
    print('#',end='')
#now building the triange:    
for i in range(5):
    for j in range(i):
        print('#',end='')
    print('#')

#building a pyramid:
for i in range(3):
    for j in range(3-i):
        print(' ',end='') 
    for j in range(i):
        print('#',end='')
    for j in range(i):
        print('#',end='')
    print('#')

#as function:
x = int(input('Define size of pyramid'))
for i in range(x):
    for j in range(x-i):
        print(' ',end='') 
    for j in range(i):
        print('#',end='')
    for j in range(i):
        print('#',end='')
    print('#')

    
#MATRIX CALCULATIONS (without numpy):
nestedList = [[j for j in range(3)] for i in range(3)]
nestedList

#how to create a matrix in python:
#inner brackets define row entries
matrix = [[1,2],[3,4]]
#to access the 1 in this matrix:
matrix[0][0]
    
#to calculate the determinant (manually wihoout numpy)
matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

#add 1 to each matrix entry:
for i in range(len(matrix)):
    for j in range(len(matrix[i])):
        matrix[i][j] = matrix[i][j] + 1
        
#using + operator:
for i in range(len(matrix)):
    for j in range(len(matrix[i])):
        matrix[i][j] += 1

matrix = [[1,2],[3,4]]     
#using nested list comprehension:
newmatrix = [[matrix[i][j]+1 for j in range(len(matrix[i]))] for i in range(len(matrix))]
newmatrix

#transpose of a matrix:
matrix = [[1,2],[3,4]]

for i in range(len(matrix)):
    for j in matrix:
        print(j[i])
#with nested list comprehension:
transpose = [[j[i] for j in matrix] for i in range(len(matrix))]
transpose
        
####PART 5####
#some info on recursion:
#a recursive function is a function calling intself during execution enabling it to repeat many times.

#factorial function:

