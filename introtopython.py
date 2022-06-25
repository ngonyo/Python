#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 17:12:17 2021

@author: nathan
"""
#NOTE: python is case sensitive
#this is a comment!
#Let's check the current working directory:
import os
os.getcwd()
print("Current working directory: {0}".format(os.getcwd()))

#to get current working directory
os.getcwd()

#to change working directory
os.chdir('/Users/nathan/Documents/Python/Datasets')
os.getcwd()

#to show contents of working directory
os.listdir()


#Note that if we define some variables, they will appear in right in "variable explorer"
x = 1, 2, 3
y = 'hello'
z = True

#We can do basic math in the code itself. lets assign some variables
a=10
b=9
(a*b)/b
a
b

print(a)
print(b)

#What are the data types in python?. lets assign each datatype to their respective name
integer = 5
floating = 3.69
stringvalue = 'hello'
boolean = True

#to print their values, simple typt the assigned variables
integer
boolean

#to find the datatypes of a variable, we can check:
type(integer)
type(stringvalue)

#lets see how different datatypes interact with each other:
w = 5
x = 3.5
y = 'hello'
z = True 
#we can add integers and floats, but not integers+strings. We can add integers or floats to booleans as booleans are assigned either 0 or 1
#we can also add strings. it just combines them

#Lets do some more with arithmatic operators:
sumwx = w+x
sumwx

#Let's assign some values for x, y amd create a histogram chart. first we must install matplotlib
import matplotlib.pyplot as plt
x = [1,2,3,4,5]
y=[2,3,5,6,2]
#here, we are assigned values to x and y as a list
type(x)
plt.bar(x,y)
plt.show()

###### PART 2#####

#Typecasting in Python
w = 5
x = 3.6
y = 'hello'
z = True

#INTEGER TYPECASTING (want to perform of typecast from datatypes to integers)
# floats to integer
int(x)
# strings to integer. this doesnt work
#int(y)
# booleans to integer. this works
int(z)

#FLOAT TYPECASTING
# integer to float. this works
float(w)

# string to float. this doesnt works
float(y)

# boolean to float. this works
float(z)

#BOOLEAN TYPECASTING
# integer to boolean. this works as any value other than 0 gives "true"
bool(w)

# float to boolean. this works
bool(x)

# string to boolean. this works
bool(y)

#INPUT FUNCTION
input()
#this will prompt the user to input something.
input('please enter something moron...')
#we can customize input prompt
x = input('please enter something for x ')
x
#we can use input function to create new functions/variables
x = int(input('please enter something for x '))
#this will guaranteee input is stored as int, so we wont habe to typecast later

#LIST INDEXING % SLICING
list1 = [2,4,6,8]
#the first element, 2 is stored at index 0. third element is stored at index 2
list1[0]
list1[2]

#we can also use negative values
list1[-1]
list1[-2]

#with indexing, syntax= listname[start:stop], where last element is last BEFORE stop
#to give us first 2 elements of list
list1[0:2]
#we can also use negative values, which count from right
list1[-4:-2]

#CONDITIONAL STATEMENTS
x = 10
if x > 5:
    print('yes, larger than 5')
    
#no else statement in above example, so if not >5, no action happens
x = 10
if x > 5:
    print('yes, larger than 5')
else:
    print('no')    

a = int(input('what value for a? '))
b = int(input('what value for b? '))

if a > b:
    print('a larger than b')
elif a == b:
    print('a equals b')    
else:
    print('a is smaller than b')    

#FOR LOOP
list2 = [2,4,6,8]
for i in list2:
    print(i+2)
    
for i in range(0,5):
    if i > 2:
        print(i)
#Notice the extra tab due to if statement being in for loop (2 loops)

#WHILE LOOPS
i = 0

while i < 10:
    i = i+1
    print(i)
    
#note same as
for i in range(1,11):
    print(i)

#with typecasting:
points = 0
while points < 10:
    points = int(input('Please provide your points: '))

###################
###### PART 3######
###################

L1 = [2,4,6,8]
L1
#Alternative way to create list: use list function. transform tuple into list
T1 = 1,2,3,4,5
L2 = list(T1)
L2

#Append function: adds item to end of a list
L2.append(6)
L2
#Remove function: remove item from list
L2.remove(6)
#Alternative way to remove items: pop
L2.pop(5)
L2
#Sort list in reverse:
L2.sort(reverse=True)

#ZIP FUNCTION: takes element from list 
names = ['Bob','Vince']
grades = ['C','A']
zip(names,grades)

list(zip(names,grades))
#merges the names and grades lists together into a new list

#Another example of zip function
listA = [1,2,3]
listB = [2,4,5]
#two lists used, so two iterators needed in for loop
for i,e in zip(listA,listB):
    print(i*e)

list3 = []
for i,e in zip(listA,listB):
    list3.append(i*e)
list3

#DICTIONARIES store values to a key value
#EX: define dictionary as just dictionary. define key value as martin.... 
#...define value assigned to martin
dictionary = {'Martin':10,'Bob':12}
type(dictionary)

#to add values to dictionary:
dictionary['Lisa'] = 15
#we can also change existing values 
dictionary['Martin'] = 20
#to remove entry from dictionary:
del dictionary['Bob']

#Zip function can be used with dictionaries as well
names = ['Bob','Vince']
grades = ['C','A']
dict(zip(names,grades))

#ITER & NEXT
x = 1,2,3
x
#create iterator function
iterator = iter(x)
next(iterator)
next(iterator)

#STRING FORMATTING
name = 'Julia'
age = 25
name+' is '+str(age)+' years old'
#achieve above easier by using format function
'{} is {} years old'.format(name,age)
#f string formatting to achieve above
f'{name} is {age} years old'

#BUILDING AN ALGORITHM
#create sigma function using for loop
n = 5
sigma = 0
for i in range(1,n+1):
    sigma = sigma + i
    print(sigma)
    
sigma

#doing this by writing a function:
def sigma(n):
    sigma = 0
    for i in range(1,n+1):
        sigma = sigma + i
    return(sigma)






