# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:30:00 2022

@author: Nathan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from scipy.stats.distributions import chi2


raw_csv = pd.read_csv('/Users/nathan/Documents/Python/Datasets/Index2018.csv')
df_comp = raw_csv.copy()
df_comp.date = pd.to_datetime(df_comp.date, dayfirst = True)
df_comp.set_index("date", inplace = True)
df_comp = df_comp.asfreq('b')
df_comp = df_comp.fillna(method = 'ffill')
df_comp['market_value'] = df_comp.ftse
del df_comp['dax'], df_comp['spx'], df_comp['nikkei'], df_comp['ftse']
size = int(len(df_comp)*0.8)
train, test = df_comp.iloc[:size], df_comp.iloc[size:]
train.head()
train.tail()
test.head()

##### AR Model Creation
#recall AR model is standardly written as Xt = C + (PHI)Xt-1 + Et where phi is less than 1 in modulus or ||
#To start with AR model creation, we must use the ACF and PACF to determine the appropriate number of lags in our model
#we will be working with the FTSE

#ACF:
sgt.plot_acf(train.market_value, zero = False, lags = 40)
plt.title("ACF for FTSE, size = 20")
plt.show()
#recall acf includes both indirect and direct lag effects, pacf only includes direct lag effects

#PACF
sgt.plot_pacf(train.market_value, lags = 40, alpha = 0.05, zero = False, method = ('ols'))
plt.title("PACF ftse", size = 24)
plt.show()

#lets create an AR(1) model
from statsmodels.tsa.arima_model import ARMA
model_ar1 = ARMA(train.market_value, order = (1,0)) #1 is number of lags, 0 means not taking residual values into consideration
results_ar1 = model_ar1.fit()
results_ar1.summary()
#Xt = C + (PHI)Xt-1 + Et
#lower part: coefficients, p values. const = C. phi1 is fitted value of phi
#we can see the constant and phi1 are statistically significant
#as our coefficents are ~0, lets add more coefficients to see if more complicated model is a better estimator

#lets create an AR(2) model
model_ar2 = ARMA(train.market_value, order = (2,0)) #1 is number of lags, 0 means not taking residual values into consideration
results_ar2 = model_ar2.fit()
results_ar2.summary()
#notice phi2 is not statistically significant / different from 0 so we will remove it from the model

#lets check AR(3) and AR(4) model
model_ar3 = ARMA(train.market_value, order = (3,0)) #1 is number of lags, 0 means not taking residual values into consideration
results_ar3 = model_ar3.fit()
results_ar3.summary()
model_ar4 = ARMA(train.market_value, order = (4,0)) #1 is number of lags, 0 means not taking residual values into consideration
results_ar4 = model_ar4.fit()
results_ar4.summary()

#LLR (Log-Liklihood Ratio Test)
#we prefer models with a higher log likelihood and a lower information critereon
#lets use the log liklihood ratio test to see if the ar3,4 models are statistically different
def LLR_test(mod_1, mod_2, DF=1):
    L1 = mod_1.fit().llf
    L2 = mod_2.fit().llf
    LR = (2*(L2-L1)) #compute test statistic
    p = chi2.sf(LR, DF).round(3) #use chi2 method and pass LR value, df as parameters, round to 3 pts
    return(p) 
#make sure to put the more simple model first, then more complicated one when comparing
LLR_test(model_ar2, model_ar3) #difference in log liklihood significant, opt for more complicated model# not we are comparing first to second
LLR_test(model_ar3, model_ar4, DF = 3)

#AR8 vs 7
model_ar7 = ARMA(train.market_value, order = (7,0)) #1 is number of lags, 0 means not taking residual values into consideration
results_ar7 = model_ar7.fit()
results_ar7.summary()
model_ar8 = ARMA(train.market_value, order = (8,0)) #1 is number of lags, 0 means not taking residual values into consideration
results_ar8 = model_ar8.fit()
results_ar8.summary()
print("\nLLR test p value = " + str(LLR_test(model_ar7, model_ar8))) #8 is not significant more efficient than ar7
#note that the ar1 model was better than the ar2 model, but higher order models all outperformed ar1



##### Stationarity and AR models
#AR models work best with stationary data
sts.adfuller(train.market_value)
#p value insigificant --> underlying data is NOT stationary
#this suggests we should not rely on an AR model to model this time series
#note that returns may be stationary - lets create a return series from ftse
train['returns'] = train.market_value.pct_change().mul(100) #mult by 100 at end for % in numbers
train = train.iloc[1:] #skip first value as no percent change for first value
sts.adfuller(train.returns)
#returns are indeed stationary

##### Creating PACF and ACF plots for return series
#ACF: recall ACF is used for MA and PACF is used for AR
sgt.plot_acf(train.returns, zero = False, lags = 40)
plt.title("ACF for FTSE returns, size = 20")
plt.show()
#consecutive values move often move in different directions --> sideways returns as expected
#PACF recall ACF is used for MA and PACF is used for AR
sgt.plot_pacf(train.returns, lags = 40, alpha = 0.05, zero = False, method = ('ols'))
plt.title("PACF ftse returns", size = 24)
plt.show()

##### AR Model for Returns
model_returns_ar1 = ARMA(train.returns, order = (1,0))
results_returns_ar1 = model_returns_ar1.fit()
results_returns_ar1.summary()           
#we can see phi1 and C are not significantly different from 0. lets try ar2

model_returns_ar2 = ARMA(train.returns, order = (2,0))
results_returns_ar2 = model_returns_ar2.fit()
results_returns_ar2.summary()     
LLR_test(model_returns_ar1, model_returns_ar2)        
#p value is less than 1% so we conclude this model is better. we can also see the p value of phi1-2 is lower

model_returns_ar3 = ARMA(train.returns, order = (3,0))
results_returns_ar3 = model_returns_ar3.fit()
results_returns_ar3.summary()     
LLR_test(model_returns_ar2, model_returns_ar3)   
##AR3 is clearly better than AR2 based on lower AIC, distintly different log likelihood *******
#in order to find the ideal AR model, we would create them until the additional coefficient is not statistically significant, LLR test fails, and higher AIC or BIC values



##### Normalizing Prices
#lets normalize two sets of data
benchmark = train.market_value.iloc[0]
train['norm'] = train.market_value.div(benchmark).mul(100)
sts.adfuller(train.norm)
#normalized prices not stationary

#Normalizing Returns:
#normalized returns account for absolute return in contrast to prices
#normalized returns allow us to compare the relative profitability as opposed to non-normalized returns
benchmark1 = train.returns.iloc[0]
train['norm_returns'] = train.returns.div(benchmark).mul(100)
sts.adfuller(train.norm_returns)
#this data is stationary
#it seems normalizing data does not change stationarity*********

##### Creating AR model for Normalized Returns
#note that when we create AR model for normalized returns, the coefficient from returns ar model change, however the coefficients will NOT change
#--> normalizing series does not effect model selection

##### Analyzing Residuals
#we determined best ar model for price series was ar7 model
train['price_residuals'] = results_ar7.resid
#lets find var , mean of residuals
train.price_residuals.mean()
train.price_residuals.var()
sts.adfuller(train.price_residuals)
#so far residuals seem to be stationary
#lets investigate further to see if residuals are white noise
#we will start by creating ACF of residuals
sgt.plot_acf(train.price_residuals, zero = False, lags = 40)
