# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:30:00 2022

@author: Nathan
"""
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.arima_model import ARMA
from scipy.stats.distributions import chi2
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

##### MA Model Creation
#general form of MA model:
    #Rt = C + (Theta)1*Et-1 + Et
    #where (Theta)1 = numeric coefficient for the value associated with 1st lag
    #Et = residuals for the current period
    #Et-1 = residuals for past period, 
    # epsiols generated after predicting each period
#MA(1) model is simular to AR(inf) model with certain restrictions
#absolute value of ThetaN is less than 1, just like Phi in AR model
#with moving average model we will rely on ACF as determining direct effects of lags on present day values is not relevant in MA model

sgt.plot_acf(train.returns, zero = False, lags = 40)
plt.title("ACF for FTSE returns, size = 20") #recall ACF is used for MA and PACF is used for AR
plt.show()

##### MA(1) Model for Returns
model_ret_ma_1 = ARMA(train.returns[1:], order = (0,1)) #first value refers to AR components, MA second
results_ret_ma_1 = model_ret_ma_1.fit()
results_ret_ma_1.summary()
#coefficient from ACF1 was not sig different from 0, so makes sense than p value of ma1 p value >.05

#MA of order 2
model_ret_ma_2 = ARMA(train.returns[1:], order = (0,2)) #first value refers to AR components, MA second
results_ret_ma_2 = model_ret_ma_2.fit()
results_ret_ma_2.summary()
#p value from error term 2 periods ago significant. also 1 period ago now significant, different from MA1 model
print("\nLLR test p value = " + str(LLR_test(model_ret_ma_1, model_ret_ma_2)))
#MA model 2 better fit based on output and LLR test

#we will keep going until we go 7 periods back as MA7 produces non sig. coefficient, fails LLR test
#recall ACF output for returns. MA8 might be a good fit as ACF8 is significant
model_ret_ma_8 = ARMA(train.returns[1:], order = (0,8)) #first value refers to AR components, MA second
results_ret_ma_8 = model_ret_ma_8.fit()
results_ret_ma_8.summary()

#we know MA8 is better than MA7. lets see if MA8 is better than MA6
#not we will have to specify difference in DF of 2 as different from default value of 1
#LLR_test(model_ret_ma_8, model_ret_ma_6, DF = 2) #commented out as I didnt run MA 3-7
#LLR test shows MA8 better than MA6



###### Examining Residuals
train['res_ret_ma_8'] = results_ret_ma_8. resid[1:]
print("Mean of Residuals is " + str(round(train.res_ret_ma_8.mean(),3)))
print("Var of Residuals is " + str(round(train.res_ret_ma_8.var(),3)))
from math import sqrt
round(sqrt(train.res_ret_ma_8.var()),3)

train.res_ret_ma_8[1:].plot(figsize = (20,5))
plt.title("Residuals of Returns", size = 24)
plt.show()
#series looks like whitenoise excluding dot com and GFC. lets use adf to verify stationarity
sts.adfuller(train.res_ret_ma_8[2:])
#finally, lets look at ACF. coefficients should not be significantly different from zero
sgt.plot_acf(train.res_ret_ma_8[2:], zero = False, lags = 40)
plt.title("ACF for residuals for returns (MA8), size = 20")
plt.show()
#notice how first ACF for 1-8 are 0 becuase these were already in our model 



##### MA Model for Normalized Returns
#normalizing input data will not change model output as was the case with AR model


##### MA Model for Price Data 
#recall AR models are less reliable when modeling non stationary data, lets see if this is true of MA models
sgt.plot_acf(train.market_value, zero = False, lags = 40)
plt.title("ACF for Prices, size = 20")
plt.show()
#based on this ACF, we would have to use an MA(inf) model to fit the data
model_ma_1 = ARMA(train.market_value, order = (0,1)) #first value refers to AR components, MA second
results_ma_1 = model_ma_1.fit()
results_ma_1.summary()
#both constant and 1 lag MA parameter clearly significant not surprisingly
#note that theta1 is almost 1(0.963) which means model tries to keep almost entire magnitude of the error term from last period
#as this is simple model with 1 lag, the error term contains all the infomation from the other lags
#as theta1 approaches zero, model approximates into AR model 

#we can conclude that MA models do not perform well for non-stationary data
#non stationary process should also include previous periods values, combining AR+MA --> ARMA model
#an ARMA model would take into account both past values AND past errors

