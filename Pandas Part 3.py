# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 18:14:35 2022

@author: Nathan
"""
import os
import pandas as pd

os.listdir()
gun = pd.read_csv('gun-violence-data_01-2013_03-2018.csv')
gun

#overview of columns:
gun.columns
#lets say we only are interested in some columns/ variables:
df = gun[['state', 'date', 'n_killed']]

#right now the dates are stored as string values, we can transform into a datetime object
pd.to_datetime(df.date)
df.date = pd.to_datetime(df.date)

#lets say we are only interested in the year in this case
df.date = df.date.dt.year
df

#GROUPING IN PANDAS
#group the frame by year
grp = df.groupby('date')
grp
#now we have created group object
#now use aggregation functions
grp.sum()
grp.sum().plot(kind = 'bar', color = 'purple')
 
#grouping by state:
grp = df.groupby('state')
grp
grp.sum() #ignore date column as it is totaling year of all instances by state
grp.sum().plot(kind = 'bar', color = 'purple')

#grouping with multiple conditions
grp2 = df.groupby(['state', 'date'])
grp2
grp2.sum()

#~indexing with group
dfx = grp2.sum()
dfx.loc['Illinois']
dfx.loc[input('which state you wanna check?')] #ask user to input state

#PIVOT TABLES
pivot = pd.pivot_table(df, values = 'n_killed', index = ['state'], columns = ['date'], aggfunc = 'sum')
pivot

#only showing certain years
pivot1 = pivot.loc[:,[2014,2015,2016,2017]]
pivot1

#state with most fatalities
mostdangerous = pivot1[2017].nlargest()
mostdangerous
mostdangerous.index 
chart = pivot1.loc[mostdangerous.index,:]
chart.plot(kind = 'bar')


