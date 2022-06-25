#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 19:07:32 2021

@author: nathan
"""

import pandas as pd
employee = ['Smith', 'Cavaliere', 'Hopkins', 'Brody']
employee

#creating dataframe
df = pd.DataFrame(employee)
df
df = pd.DataFrame(employee, columns=['employee'])
df

#adding new variable / column
worktime = [40,35,25,50]
df['worktime'] = worktime
df

#in one step:
df = pd.DataFrame({'employee': employee, 'worktime': worktime})
df

import os
#to get current working directory
os.getcwd()

#to change working directory
os.chdir('/Users/nathan/Documents/Python/Datasets')
os.getcwd()

#to show contents of working directory
os.listdir()

#reading files with pandas
df_csv = pd.read_csv('cereal.csv')
df_csv

#one line:
df_csv = pd.read_csv('/Users/nathan/Documents/Python/Datasets/cereal.csv')

#we can also use python built in function, but this often does not work as easy as pandas
#ex: file = open('cereal.csv')
#file.read()



#SELECTING DATA IN PANDAS
import pandas as pd
#we will be using cereal dataset, containing 80 entries
dfcereal = pd.read_csv('cereal.csv')
dfcereal.head()
dfcereal.tail()

#descrobe function: summary statistics of dataset
dfcereal.describe()

#get individual columns, based on headers:
df['name']
df['type']

#Note we can show the data as either a series or dataframe, these are different
#as dataframe:
df[['name']]
type(df[['name']]) 
#as series:
df['name']
type(df['name'])

#selecting multiple columns:
df[['name', 'protein', 'fat']]

#selecting rows:
#to get first 10 rows
df[0:10]

#using the df.loc and df.iloc function:
df.iloc[0:13]
df.loc[0:14]
#note 10th row is included because loc function reads 10 as stopping point

#filtering by both columns and rows:
df.iloc[0:10,0:2]
#this gives us data frame of first 10 rows and first 2 columns

#again, the difference between df.loc and df.iloc
df.iloc[0:10, 0:2]
df.loc[0:10,'name','protein']
#3For label selection, df.loc | for value selection, use df.iloc


#FILTERING DATA IN PANDAS
df = pd.read_csv('cereal.csv')
#boolean indexing:
#first, we must create a boolean mask. this evaluates each entry of dataframe against boolean argument
df['protein'] >= 4
#as new df, with boolean mask as arguemnt
df[df['protein'] >= 4] #this is called boolean indexing

#multiple filters: high protein and low sugar
(df['protein'] >= 4) & (df['sugars'] < 10)

df[(df['protein'] >= 4) & (df['sugars'] < 10)]

#every logical operator can be used here: & (and), | (or), ~ (not)

#ex:high protein or low fat
df[(df['protein'] >= 4) | (df['fat'] < 2)]