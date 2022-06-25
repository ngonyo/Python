# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 19:11:27 2022

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
from arch import arch_model 

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

train['returns'] = train.market_value.pct_change().mul(100)
train['sq_returns'] = train.returns.mul(train.returns) 



##### GARCH (Generalized Auto Regressive Conditional Heteroscedastic) Model Form
#General Form: var(yt | yt-1) = (Omega) + (alpha1)E^2t-1 + (Beta1)Sigma^2t-1
    #Omega = constant. from ARCH constant was alpha0. here can have 2 variables in constant - squared residuals and conditional variance
    #(alpha1)E^2t-1 = squared residual from last period * coefficient alpha same as ARCH
    # (Beta1)Sigma^2t-1 conditional variance from last period * coefficient
    
#GARCH has two orders: 
    #First: ARCH Component: E^2t = past squared residual
    #Second: GARCH Component: Sigma^2t = past conditional variances
    #this form is simular to ARMA, where AR component is simular to GARCH comp, MA comp ~= ARCH comp
#there are ARMA - GARCH models where the mean component is an ARMA(p,q) and variance component is a GARCH(p,q) 


##### Fitting GARCH(1,1) with serially uncorrelated mean
#serial uncorrelated mean --> mean doesnt rely on past values or errors. constant mean model
model_garch_11 = arch_model(train.returns[1:], mean = "Constant", vol = "GARCH", p = 1, q = 1)
results_garch_11 = model_garch_11.fit(update_freq = 5)
results_garch_11.summary()    

##### Higher Order GARCH models
#Note that literature on the topic has shown that no higher order GARCH models outperform GARCH(1,1) when it comes to variance of market returns
#this is a result of the recursive nature in which past variances are computed
#all the conditional variance 2 days ago will be contained in the conditonal variance of yesterday
#--> no need to include more than 1 GARCH component

#note that in the arch_model function p and q are flipped

model_garch_12 = arch_model(train.returns[1:], mean = "Constant", vol = "GARCH", p = 1, q = 2)
results_garch_12 = model_garch_12.fit(update_freq = 5)
results_garch_12.summary()   
#note that p value of beta coefficient is exactly 1, showing perfect multicolinearity due to the relationship between conditional variances
#in other words, all the explanatory power of the conditional variance two periods ago is already captured by the variance from last period.

#lets now look at GARCH(2,1)
model_garch_21 = arch_model(train.returns[1:], mean = "Constant", vol = "GARCH", p = 2, q = 1)
results_garch_21 = model_garch_21.fit(update_freq = 5)
results_garch_21.summary()  
#we can see alpha 1, the additonal coeff is not SS at 5% alpha

























