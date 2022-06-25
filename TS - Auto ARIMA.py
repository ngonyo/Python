# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 11:53:06 2022

@author: Nathan
"""

import pandas as pd
import numpy as np
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from statsmodels.tsa.arima_model import ARIMA
from arch import arch_model 
import yfinance
import warnings
warnings.filterwarnings("ignore")
sns.set()

raw_data = yfinance.download (tickers = "^GSPC ^FTSE ^N225 ^GDAXI", start = "1994-01-07", end = "2018-01-29", interval = "1d", group_by = 'ticker', auto_adjust = True, treads = True)
df1 = raw_data.copy()
df1['spx'] = df1['^GSPC'].Close[:]
df1['dax'] = df1['^GDAXI'].Close[:]
df1['ftse'] = df1['^FTSE'].Close[:]
df1['nikkei'] = df1['^N225'].Close[:]
df1 = df1.iloc[1:]
df1=df1.asfreq('b')
df1 = df1.fillna(method = 'ffill')

df1['ret_spx'] = df1.spx.pct_change(1)*100
df1['ret_ftse'] = df1.ftse.pct_change(1)*100
df1['ret_dax'] = df1.dax.pct_change(1)*100
df1['ret_nikkei'] = df1.nikkei.pct_change(1)*100

size = int(len(df1)*0.8)
df, dftest = df1.iloc[:size], df1.iloc[size:]

#Recall that normall we want models with a lower AIC and BIC
#Mathematically, AIC = -2(lnL/T)+k(2/T)
                #BIC = -2(lnL/T)+k(lnT/T)
    #where lnL = log liklihood of estimated model, k = # of parameters T = length of time series
    
##### Auto Arima
#Auto Arima cycles through different forms of MA, AR, ARIMA and returns one with lowest AIC
#this method saves time and need for compare models manually. it also reduces the risk of human error
#the downside of this model is that it puts all emphasis on AIC. we also dont get to see what models were "runners up" and doesnt factor in possibility of overfitting
#make sure to specify correct parameters. in our ARCH modeling the default was GARCH

from pmdarima.arima import auto_arima

model_auto = auto_arima(df.ret_ftse[1:]) #lets see what default model parameters are
model_auto
model_auto.summary() #*******interpretation of Q and JB test statistics
#note that SARIMAX includes seasonality, AR, integration, MA, exogenous  factors so SARIMAX will be default but some parameters might = 0 as is here
#this is an ordinary ARMA disguised as SARIMAX. this is an ARMA(4,5) as integration part is 0
#notice how sample goes from 0 to 5020, which states we are using all of sample to build model as it had 5020 time points
#*in vs out of sample split
#auto arima takes complete dataset and automatically splits it up, validates model by itself
#note that we selected a different ARIMA model before, as we considered characteristics other than AIC


##### Fine Tuning Arguments of AutoARIMA function
# arguemnts summary:
# exogenous -> outside factors (e.g other time series) 
# m -> seasonal cycle Length 
# max_order -> maximum amount of variables to be used in the regression (p + q) 
# max_p -> maximum AR components 
# max_q -> maximum MA components 
# max_d -> maximum Integrations 
# maxiter -> maximum iterations we're giving the model to converge the coefficients (becomes harder as the order increases) 
# return_valid fits -> whether or not the method should validate the results 
# alpha -> Level of significance, default is 5%, which ive should be using most of the time 
# njobs -> how many models to fit at a time (-1 indicates "as many as possible 
# trend -> "ct" usually 
# information_criterion -> 'aic', 'aicc', 'bic', 'hqic", "oob (Akaike Information Criterion, corrected Araike Information Criterion Bayesian Information Criterion Hannan-Quinn Information Criterion, or "out of bag"--for validation scoring--respectively)
# out of sample size -> validates model select (pass entire dataset and set 20% to be out of sample size)

#lets create an auto arima model and use exogenous variable of sp500 returns
model_auto = auto_arima(df.ret_ftse[1:], exogenous = df['ret_spx'])
model_auto
model_auto.summary()
#adding multiple exogenous variables and specying length of season 's' which is denotes m here # m -> seasonal cycle Length 
#we will set m to 5 as 5 business days per week
#lets also set total number of non seasonal ar and ma components model can have using max_order none (sum) but setting each individual to 7
#lastly, lets set max number of seasonal orders P,Q,D. defaults are 2,2,1
model_auto = auto_arima(df.ret_ftse[1:], exogenous = df[['ret_spx', 'ret_dax', 'ret_nikkei']][1:], m = 5, max_order = None, max_p = 7, max_q = 7, max_d = 2, max_P = 4, max_Q = 4, max_D = 2) 
model_auto
model_auto.summary()


#more info on model parameters:
    
#note that sometimes complicated models fail to converge if models have too many exogenous variables or too high of a total order
    #we can address this by increasing maxiter
#n_jobs indicates how many CPU cores should be utilized / how many models to fit at a time. -1 indicates starting at max which may slow down pc
#trend: specify endogenous variables presence of trend. we set trend = ct for constant and trend. ctt = quadratic relationship. 
    #can also use boolean values here. ie [1,0,0,1] indicates constant term and trend of third degree
#we dont have to evaluate models just on AIC. we use information_criterion
    #if using oob or out of bag we must seperate data into in sample, out of sample sets
    
#setting out of sample size and using OOB instead of AIC example:
model_auto = auto_arima(df.ret_ftse[1:], exogenous = df[['ret_spx', 'ret_dax', 'ret_nikkei']][1:], information_criterion = "oob", out_of_sample_size = 1000) 
#this will use the last 1000 observations as a training set
#we can also automatically set it to 20% of dataset as follows:
model_auto = auto_arima(df.ret_ftse[1:], exogenous = df[['ret_spx', 'ret_dax', 'ret_nikkei']][1:], information_criterion = "oob", out_of_sample_size = int(len(df1)*0.2)) 
#if we do above method we have to make sure to set the dataset to full dataset as opposed to just training as already splitting in validation
model_auto = auto_arima(df1.ret_ftse[1:], exogenous = df1[['ret_spx', 'ret_dax', 'ret_nikkei']][1:], information_criterion = "oob", out_of_sample_size = int(len(df1)*0.2)) 
model_auto
model_auto.summary()
    
