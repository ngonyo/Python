#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 16:13:40 2022

@author: nathan
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import os
#to get current working directory
os.getcwd()

#to change working directory
os.chdir('/Users/nathan/Documents/Python/Datasets')
os.getcwd()

#to show contents of working directory
os.listdir()

#reading files with pandas
df = pd.read_csv('cereal.csv')

#Descrptive statistics
df.describe()

#note there are negative valyues for sugar. lets say we want to access these entries
#using boolean indexing:
(df['carbo'] == -1) | (df['sugars'] == -1) | (df['potass'] ==-1)
#showing of these rows:
df[(df['carbo'] == -1) | (df['sugars'] == -1) | (df['potass'] ==-1)]
#using indexing to show rows:
df[(df['carbo'] == -1) | (df['sugars'] == -1) | (df['potass'] ==-1)].index
# we can see rows 4, 20 and 57 contain negative values (error)

#now we will delete these 'outlier' rows
delete = df[(df['carbo'] == -1) | (df['sugars'] == -1) | (df['potass'] ==-1)].index
df.drop(delete)

#storing as previous dataframe:
df = df.drop(delete)
df

#simple descriptive statistics
df.describe()

#creating correlation matrix
df.corr()

#creating heatmap of correlation matrix
sns.heatmap(df.corr())
plt.show() #to call plot in file

#pairwise correlation
sns.relplot(x = 'rating', y = 'sugars', data = df)
#we can see the lower the rating, the higher the sugar

#lets compare the cereals macroingredients
macros = df[['protein', 'fat', 'carbo', 'sugars']]
macros

#boxplot to understand disribution of data
macros.boxplot()
plt.show()
