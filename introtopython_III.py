#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 21:36:02 2021

@author: nathan
"""
####PART 5####
#some info on recursion:
#a recursive function is a function calling intself during execution enabling it to repeat many times.

#factorial function:
    
def factorial(n):
    if n == 0:
        return 1
    else:
        return factorial(n-1) * n
factorial(5)

#sigma function:
def sigma(n):
    if n == 0:
        return 0
    else:
        return sigma(n-1) + n

sigma(5)

##EXAMPLE OF LINEAR REGRESSION IN PYTHON
import matplotlib.pyplot as plt
import statsmodels.api as sm
import umpy as np

#variables: analysis = time spent on data analysis tasks in company
#variables: seniority = tyears employee worked at company
Analysis = (10, 8, 6, 8, 12, 18, 15)
Seniority = (12, 14 ,20 ,19 ,7, 3, 5)
plt.scatter(x=Seniority, y=Analysis)
plt.show()

#with labels:
plt.scatter(x=Seniority, y=Analysis)
plt.xlabel('Seniority')
plt.ylabel('Hours per week')
plt.title('Weekly hours spent on data analysis vs years of seniority')
plt.show() 

#simple linear regression using numpy
z = np.polyfit(Seniority, Analysis, 1) #1 signifies linear function, 2 would signify 
p = np.polyld(z)
print(z)
print(p)
plt.plot(Seniority, p(Seniority))
plt.show

#simple linear regression using statsmodels
y = Analysis
x = Seniority
x = sm.add_constant(x)
model = sm.OLS(y,x) #OLS spedifys ordinary least squares model
results = model.fit()
results.summary()
#we can see both packages returned same regression output

#lambda functions:
def multiply(x):
    return x*2
#performed with lambda function:
(lambda x: x*2)(5)
#in another function:
lam = lambda x: x*2
#check if value meets a specified condition:
(lambda x: x > 5)(7)

#multiplying two numbers:
def multiply2(x,y):
    return x*y
#using lambda function:
(lambda x,y: x*y)(5,5)

#with if/else added in:
def multiply3(x,y):
    if x > y:
        return x*y
    else:
        return x-y
#using lambda function:
(lambda x,y: x*y if x>y else x-y)(5,5)

#MULTIPLE KEYS FOR VALUES IN A DICTIONARY:
grades = {'Sara':'A','Joe':'B','Vincent':'C','Martin':'D','Lisa':'A'}
grades
#to get key value:what grade did a student get?
grades['Sara']
#other way around (find students that got certain grade)
#using one iterator:
    
Astudents = []
for i in grades:
    if grades[i] == 'A':
        Astudents.append(i)        
Astudents
#using list comprehension:
[i for i in grades if grades[i] == 'A']

#using two iterators:
grades.items()

Astudents = []
for i,e in grades.items():
    if e == 'A':
        Astudents.append(i)

#using list comprehension:
[i for i, e in grades.items() if e == 'A']

    
#MAP FUNCTION

    
    
































