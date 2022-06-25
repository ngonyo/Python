#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 18:39:51 2021

@author: nathan
"""

#linear regression
import pandas as pd
df = pd.read_csv('data.csv')
df = df[df.Gender == 'Female']
df

x = df.Weight
y = df.Height

import matplotlib.pyplot as plt
plt.scatter(x,y)

#lets calculate slope of regression line, ie Beta1 in y = beta1*x + beta0
#we must calculuate SSxx and SSxy
#slope = SSxy/SSxx

xmean=x.mean()
xmean

#calculating SSxx: SSxx = Sigma(xbar - x)^2
df['diffx'] = xmean - x
df
df['diffx_squared'] = df.diffx**2
SSxx = df.diffx_squared.sum()

#calculating SSxy: SSxy = Sigma(xbar - x)*(ybar - y)
ymean = y.mean()
df['diffy'] = ymean-y
SSxy = (df.diffx * df.diffy).sum()

beta1 = SSxy/SSxx
beta1
#calculating intercept beta0 of regression line
#beta0 = ybar - beta1*xbar
beta0 = ymean - beta1*xmean

#predicting values:
def predict(value):
    predict = beta1*value+beta0
    return predict
    
predict(150)    

#adding regression line to scatter plot:
plt.scatter(x,beta1*x+beta0, 'r')   


#sklean comparison:
#comparing last model
beta0
beta1
from sklearn import linear_model

x = df.Weight[['Weight']]
y = df.Height
model = linear_model.LinearRegression()


model.fit(x,y)
model.coef_
model.intercept_
model.predict([[150]])
