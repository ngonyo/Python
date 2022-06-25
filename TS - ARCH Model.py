# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 16:58:05 2022

@author: Nathan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats.distributions import chi2
from math import sqrt
import seaborn as sns

raw_csv = pd.read_csv('/Users/nathan/Documents/Python/Datasets/Index2018.csv')
df_comp = raw_csv.copy()
df_comp.date = pd.to_datetime(df_comp.date, dayfirst = True)
df_comp.set_index("date", inplace = True)
df_comp = df_comp.asfreq('b')
df_comp = df_comp.fillna(method = 'ffill')
df_comp['market_value'] = df_comp.ftse
size = int(len(df_comp)*0.8)
train, test = df_comp.iloc[:size], df_comp.iloc[size:]

def LLR_test(mod_1, mod_2, DF=1):
    L1 = mod_1.fit().llf
    L2 = mod_2.fit().llf
    LR = (2*(L2-L1)) #compute test statistic
    p = chi2.sf(LR, DF).round(3) #use chi2 method and pass LR value, df as parameters, round to 3 pts
    return(p) 


#ARCH Model Form
#ARCH model is most common approach to modeling volatility or variance in a time series
#ARCH is NOT used to forcast future values, only vol
#ARCH model consists of two equations, one for the mean and one for variance to model expected value of the series and vol.

#ARCH stands for Autoregressive Conditional Heteroskedastic Model
#consider simple ARCH model:
    #mean equation: mt = c0 + (phi1)mt-1
    #variance equation: var(yt|yt-1) = alpha0 + (alpha1)E^2t-1
    #notice how variance equation depends on previous variance, squared epsilon
#higher order ARCH models:
    #ARCH(q) is number of previous values we include in model (AR) component
#mean in ARCH model could be constant, a time series, or modeled based on other model such as ARMAX *****    


#creating returns:
train['returns'] = train.market_value.pct_change().mul(100)
#creating squared returns
train['sq_returns'] = train.returns.mul(train.returns)    
    
#returns vs squared returns:
train.sq_returns.plot(figsize=(20,5))
plt.title("Volatility", size = 24)
plt.show()
    
#Lets look at PACF of returns and squared returns
sgt.plot_pacf(train.returns[1:], lags = 40, alpha = 0.05, zero = False, method = ('ols'))
plt.title("PACF of Returns", size = 20)
plt.show()   
sgt.plot_pacf(train.sq_returns[1:], lags = 40, alpha = 0.05, zero = False, method = ('ols'))
plt.title("PACF of Squared Returns", size = 20)
plt.show()   
#based on this PACF plot, there seems to be short term trends in variance / volatility
    

#Fitting an ARCH model to dataset
from arch import arch_model    
model_arch_1 = arch_model(train.returns[1:])
results_arch_1 = model_arch_1.fit()
results_arch_1.summary()    
#constant mean implies constant mean rather than dynamic, which makes sense for returns
#vol model GARCH confirms we are using GARCH model for variance equation
#below we see residuals norm. distributed
#method dictates how we find coefficients of model
#notice how degrees of freedom is 4 

#volatility model interpretation:
    #var(yt|yt-1) = alpha0 + (alpha1)E^2t-1
#omega represents constant value in variance / alpha0
#alpha1 represnts alpha1
#beta1 will be discussed further later below
    
#Iteration interpretation
#notice above the parameter output model has several iterations before it converges. this is due to fitting two equations at same time
#process: fits model with coefficients, checks performance, adjusts, ... stops when LL decreases or absolute value increases if negative
#to avoid seeing all info for each iteration, adjust as follows:
model_arch_1 = arch_model(train.returns[1:])
results_arch_1 = model_arch_1.fit(update_freq = 5)
results_arch_1.summary()     
    


##### The Simple ARCH(1)
#this assumes mean of series is not serially correlated, not time invariant
#we will specify constant mean approach
#lets also specify variance method of ARCH
#we will also specify order of model, with p parameter *******
model_arch_1 = arch_model(train.returns[1:], mean = "Constant", vol = "ARCH", p =1)
results_arch_1 = model_arch_1.fit(update_freq = 5)
results_arch_1.summary()     
#notice how R squared is almost 0. R squared is a measurement of explanatory variation away from the mean
#if residuals are simply a version of original dataset, where each value is decreased by a constant, then there will be not actual variance to explain
#R squared is not useful in interpretation for ARCH models    
#3 degrees of freedom: 3 parameters are c0 in mean (constant) and alpha0, alpha1 in variance equation

#we could change mean argument to "AR" or "MA" if we wished
#ex: assuming we want mean to be modeled with AR model with lags of order 2,3,6
model_arch_1 = arch_model(train.returns[1:], mean = "AR", lags = [2, 3, 6], vol = "ARCH", p =1)
results_arch_1 = model_arch_1.fit(update_freq = 5)
results_arch_1.summary()     

#we can also set different probability disributions for the error terms such as "t" or "ged" for students t or generalized error dist.
model_arch_1 = arch_model(train.returns[1:], mean = "AR", lags = [2, 3, 6], vol = "ARCH", p =1, dist = "t")
results_arch_1 = model_arch_1.fit(update_freq = 5)
results_arch_1.summary()     



##### Higher-Lag ARCH Model
model_arch_2 = arch_model(train.returns[1:], mean = "Constant", vol = "ARCH", p =2)
results_arch_2 = model_arch_2.fit(update_freq = 5)
results_arch_2.summary()  

model_arch_3 = arch_model(train.returns[1:], mean = "Constant", vol = "ARCH", p =3)
results_arch_3 = model_arch_3.fit(update_freq = 5)
results_arch_3.summary()  

#we can keep increasing number of past residuals being taken into account until LL stops going up or coefficients become non SS or AIC increases


    