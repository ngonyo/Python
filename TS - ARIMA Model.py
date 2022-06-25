# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 15:34:23 2022

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
del df_comp['dax'], df_comp['spx'], df_comp['nikkei'], df_comp['ftse']
size = int(len(df_comp)*0.8)
train, test = df_comp.iloc[:size], df_comp.iloc[size:]
train.head()
train.tail()
test.head()

def LLR_test(mod_1, mod_2, DF=1):
    L1 = mod_1.fit().llf
    L2 = mod_2.fit().llf
    LR = (2*(L2-L1)) #compute test statistic
    p = chi2.sf(LR, DF).round(3) #use chi2 method and pass LR value, df as parameters, round to 3 pts
    return(p) 

train['returns'] = train.market_value.pct_change().mul(100)

##### ARMA Model Form
#ARIMA model is simular to ARMA model, but behaves better with stationary data
#ARIMA model generally has three parameters in form (p , d, q)
    #p: AR component, q: MA component, d: integration parameter
    #d represents number of times we need to integrate the time series to ensure stationarity
#ARIMA(p, 0, q) ~ = ARMA(p, q)
#integrated models account for non seasonal differences between periods
#ARIMA(p, d, q) is an arma model for a newly generated stationary time series, which requires n integrations to be stationary
#an integration of order 2 would be the series resulting from integrating an order 1 series (differences in differences of prices)
#like ARMA, ARIMA does not have simple function such as ACF of PACF to suggest optimal parameter
    #instead, we will simply examine the ACF of the residuals to get a better idea of how to change the model
#note that each integration will result in a loss of observation
#ARIMA(1,1,1) model for price series
    #DeltaPt = c + (Phi1)(DeltaPt-1) + (Theta1)(Et-1) + Et



#Fitting simple ARIMA model for Price Series
model_ARIMA_1_1_1 = ARIMA(train.market_value, order = (1,1,1))
results_ARIMA_1_1_1 = model_ARIMA_1_1_1.fit()
results_ARIMA_1_1_1.summary()
#note there are only two coefficients as integration order (d) has no effect on number of parameters we need to estimate
#residuals of ARIMA(1,1,1)
train['res_ARIMA_1_1_1'] = results_ARIMA_1_1_1.resid
sgt.plot_acf(train.res_ARIMA_1_1_1, zero = False, lags = 40)
plt.title("ACF of Residuals for ARMA(1,1,1)", size = 20)
plt.show()
#we need to remove first row of data, or we can use residuals from the second period onward, which we will do below
train['res_ARIMA_1_1_1'] = results_ARIMA_1_1_1.resid
sgt.plot_acf(train.res_ARIMA_1_1_1[1:], zero = False, lags = 40)
plt.title("ACF of Residuals for ARMA(1,1,1)", size = 20)
plt.show()
#we should incorportate lags from 3rd,4th period into model based on ACF plot

#make sure to include enough start AR lags, give numerical values greater than p to start_ar_lags argument if crashing
#To determine best fitting models, we will look at LLF and AIC values as before in ARMA model
#be sure to run LLR test when comparing nested models to see if higher complexity is SS. better
#dont forget to adjust for the number of parameters / degrees of freedom difference in the LLR function
#if model looks like a good fit based on p values, LLF, AIC, LLR, then examine AIC chart to see which lag periods are statistically sign. as there might stll yet be better models
#dont forget going too far back in lagged periods for AR or MA can result in overfitting, lack of predictive power



#Models with Higher Orders of Integration
#if single layer of integration (d=1) accomplishes stationarity, additional orders of integration are unnecessary
#how do we know if an integrated dataset is stationary?
    #1 - manually create integrated version of original time series
    #2 - use ADF test

#lets integrate price series to integration of order 1
#we will create new variable to store delta in prices
train['delta_prices'] = train.market_value.diff(1) #1 indicates differences between values 1 period apart
sts.adfuller(train.delta_prices[1:]) #ignore first row
#adf test suggests stationarity, additional layers of integration not needed

#issues with ARIMA models compared to ARMA models
#more computationall expensive. more layers means more background computation, differencing when integrating
    #we must transform the data several times, differentiate the values from zero
    #more layers means harder it is to intepret the results
    
#******** why refer to integrated price series as integrated isnt it more like a derivative as it is concerning differences between time periods ie rate of change?












































