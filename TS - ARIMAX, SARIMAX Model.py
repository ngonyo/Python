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

##### ARIMAX Model Form
#MAX models take into account more than just past prices or residuals
#ARMAX - non integrated
#ARIMAX - integrated
#ARIMAX form:
    #DeltaPt = c + BetaX + (Phi1)(DeltaPt-1) + (Theta1)(Et-1) + Et
#ARMAX form:
    #Pt = c + BetaX + (Phi1)(Pt-1) + (Theta1)(Et-1) + Et
#X can be any variable we are interested in such as:
    #time varying measurement: interest rate, gas price etc
    #categorical variable ie day of week
    #boolean value accounting for festival / holiday season
    
#we need to specify the exogenous argument called "exog"
#lets use sp500 values as exogenous argument in our ftse model
model_ARIMA_1_1_1_Xspx = ARIMA(train.market_value, exog = train.spx, order = (1,1,1))
results_ARIMA_1_1_1_Xspx = model_ARIMA_1_1_1_Xspx.fit()
results_ARIMA_1_1_1_Xspx.summary()
#model suggest spx as exogenous variable is not statistically significant

#******** why refer to integrated price series as integrated isnt it more like a derivative as it is concerning differences between time periods ie rate of change?

##### SARIMAX Model Form
#SARIMAX takes into account seasonality of data ex factoring in monthly data when predicting christmas tree sales
#SARIMAX has more parameters SARIMAX(p,d,q)(P, D, Q ,s) 
#first three orders are seasonal variations of ARIMA orders
    #P: seasonal AR order
    #D: seasonal integration order 
    #Q: seasonal MA order
    #s: number of periods needed to pass before tendency reappears (cycle length)
        #if s is set to 1, that implies no seasonality, just autocorrelation

#Ex: SARIMAX(1,0,2)(2,0,1,5)
#this would include the lagged value from 5, 10 and error term from 5 periods ago
#Yt-5 = Phi1, Yt-10 = Phi2, Et-5 = Theta1
#for lagged periods, we are interested in S, up to S * P which here = 5 * 2
#in Sarimax model, total number of coefficients = sum of seasonal and non seasonal AR orders. P + Q + p + q = 6
#seasonal components are expressed in uppercase Phi, Theta to distinguish from non seasonal components

#implementing above SARIMAX model
from statsmodels.tsa.statespace.sarimax import SARIMAX
model_sarimax = SARIMAX(train.market_value, exog = train.spx, order = (1,0,1), seasonal_order = (2,0,1,5))
results_sarimax = model_sarimax.fit()
results_sarimax.summary()

#****** ask hanson for help with interpretation here









































