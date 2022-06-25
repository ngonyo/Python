# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 17:57:51 2022

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

##### ARMA Model Form
#ARMA model takes form Rt = C + (Phi1)Rt-1 + (Theta)1Et-1 + Et
#where (Phi)i denotes what part of the value from ith period is relevant in explaining current period
#(Theta)i denotes what part of the error in predicting ith period is relevant in explaining current period
#an ARMA model combines the AR model (past values) AND MA model (past errors)
#both theta and phi will be less than 1 in modulus



##### ARMA(1,1) Model for Returns
model_ret_arma11 = ARMA(train.returns[1:], order = (1,1))
results_ret_arma11 = model_ret_arma11.fit()
results_ret_arma11.summary()
#we can see only the constant is not stat. sig. different from 0
#AR coeff is positive, denoting positive tendency between past and present values
#MA is negative, suggesting we would be moving away from past period values
    #this can be interpreted that we want to prevent our targets from moving before they are accounted for in the model

#Lets see if ARMA(1,1) is a better model than AR1 or MA1
model_ret_ar_1 = ARMA(train.returns[1:], order = (1,0))
model_ret_ma_1 = ARMA(train.returns[1:], order = (0,1))
print("\nARMA vs AR ", LLR_test(model_ret_ar_1, model_ret_arma11)) #more complicated model second
print("\nARMA vs MA ", LLR_test(model_ret_ma_1, model_ret_arma11))
#p value 0 for both LLR tests - ARMA model performed better



##### Higher Lag ARMA Models
#consider the ACF, PACF plots. recall ACF is used for MA and PACF is used for AR
sgt.plot_acf(train.returns, zero = False, lags = 40)
sgt.plot_pacf(train.returns, lags = 40, alpha = 0.05, zero = False, method = ('ols'))
#plots suggest prefered ARMA would contain no more than 6 autoregressive terms, 8 MA terms
#if AR6, MA8 were able to explain changes on their own, then using them both would be redundant
#in an ARMA(8,6) model, many of the coefficients would cancel each other out
#output for this model showed more than half of the coefficients were not statistically significant
#because of this, lets start with a little less than half as many terms --> ARMA3,3

model_ret_arma33 = ARMA(train.returns[1:], order = (3,3))
results_ret_arma33 = model_ret_arma33.fit()
results_ret_arma33.summary()
LLR_test(model_ret_arma11, model_ret_arma33, DF = 4) #looks like ARMA33 is better as LLR pval = 0
#notice how we need df to be set to 4 as comparing model with 6 vs 2 param.
#first lag ma coeff not S.S. different from 0
#******* ask about rooots in model output


##### Fine Tuning Model Parameters
#we expect optimal model to be between ARMA(1,1) and ARMA(3,3)

#we will start with ARMA(3,2) then look at ARMA(2,3)
model_ret_arma32 = ARMA(train.returns[1:], order = (3,2))
results_ret_arma32 = model_ret_arma32.fit()
results_ret_arma32.summary()
#we cam see all coeff S.S.
#as lag increases, absolute value for ar and ma coeffs decrease, meaning farther back in time we go the less relevant the lagged values and errors become
#note that if we were to compare ARMA(3,3) and ARMA(3,2) a higher LLR wouldnt necessarily mean it is better as 3,3 model results in non significant coefficients

#now looking at ARMA(2,3)
model_ret_arma23 = ARMA(train.returns[1:], order = (2,3))
results_ret_arma23 = model_ret_arma32.fit()
results_ret_arma23.summary()
#one ma coefficient is not statisticall sig, we will avoid using this model

#now lets look at ARMA(3,1)
model_ret_arma31 = ARMA(train.returns[1:], order = (3,1))
results_ret_arma31 = model_ret_arma31.fit()
results_ret_arma31.summary()
#we cam see all coeff S.S.
#we can also see ma coeff is + and all ar coeff -, which makes sense from financial point of view (bouncing market)

#ARMA(2,2)
model_ret_arma22 = ARMA(train.returns[1:], order = (2,2))
results_ret_arma22 = model_ret_arma22.fit()
results_ret_arma22.summary()
#notice how both coeff associated with second lag not S.S. which also is last one.
#simpler models like the 1,2 or 2,1 would likely outperform 2,2 so we will avoid 2,2

#Finally, ARMA(1,3)
model_ret_arma13 = ARMA(train.returns[1:], order = (1,3))
results_ret_arma13 = model_ret_arma13.fit()
results_ret_arma13.summary()
#we cam see all coeff S.S.

#usually we would go on to use LLR test to compare to ARMA(3,2)
#However, 3,2 and 1,3 arent "nested"
#any AR(P) model is nested in AR(P+1)

#When are models equal:
#AR(P) = ARMA(P, 0)
#MA(Q) = ARMA(0, Q)

#When are models nested: ************
#consider ARMA(p1, q1) and ARMA (p2, q2) are nested IFF
    #1. P1 + Q1 > P2 + Q2
    #2. P1 >= P2
    #3. Q1 >= Q2
#due to third condition not being met, LLR test is void as 3,2 and 1,3 arent nested
#because of this, we manually compare the log-likelihoods and AICs of both models (want higher LLR, lower AIC)
print("\n ARMA(3,2): LL = ", results_ret_arma32.llf, "\tAIC = ", results_ret_arma32.aic) 
print("\n ARMA(1,3): LL = ", results_ret_arma13.llf, "\tAIC = ", results_ret_arma13.aic)
#>>ARMA(3,2) better than ARMA(1,3)
#Best model is ARMA(3,2) as 1. all significant coefficients 2. outpredicts all less complex alternatives

#our general process will be as follows
#begin with over parameterized model, find better, simpler option, comparing less complex models to each other



##### Analyzing Residuals for Returns
train['residuals_return_ARMA_3_2'] = results_ret_arma32.resid[1:]
train.residuals_return_ARMA_3_2.plot(figsize = (20,5))
plt.title("Residuals of Returns with ARMA(3,2", size = 24)
plt.show()
#lets look at the residual acf plot
sgt.plot_acf(train.residuals_return_ARMA_3_2[2:], zero = False, lags = 40)
#it seems that accounting for returns or residuals 5 periods ago could improve our predictions

##### Going Back and Looking at Lag5 ARMA Models
#We will now 1) look at ARMA(5,5), ARMA(5,Q), ARMA(P,5) models, 
    #2) then run the LLR test of nested ones 3) manually compare the LL and AIC values for non nested ones
    #we would have to look at ARMA(5,(1,2,3,4)) and ARMA((1,2,3,4),5) and ARMA(5,5)
    #after doing this in video, best model of this group is ARMA(5,1)
model_ret_arma51 = ARMA(train.returns[1:], order = (5,1))
results_ret_arma51 = model_ret_arma51.fit()
results_ret_arma51.summary()
    #(5,1) is also better than our previously best model, 3,2 based on AIC and LL
print("\n ARMA(3,2): LL = ", results_ret_arma32.llf, "\tAIC = ", results_ret_arma32.aic) 
print("\n ARMA(5,1): LL = ", results_ret_arma51.llf, "\tAIC = ", results_ret_arma51.aic)
#********* ll always - or no? if so why?

##### Analyzing Residuals of New Model
train['residuals_return_ARMA_5_1'] = results_ret_arma51.resid[1:]
sgt.plot_acf(train.residuals_return_ARMA_5_1[2:], zero = False, lags = 40)
#not only is the 5th lag no significant anymore, but neither are any else up until 18. we wont go this far back to avoid overfitting



##### How do ARMA Models Perform with Prices (Non Stationary Data)
sgt.plot_acf(train.market_value, zero = False, lags = 40)
plt.title("ACF for Prices, size = 20")
plt.show()
sgt.plot_pacf(train.market_value, zero = False, lags = 40)
plt.title("PACF for Prices, size = 20")
plt.show()
#note that this ACF suggests using an MA(inf) model, which is = to a simple MA1 model
#therefore, as long as we include AR components, we should be able to describe the data well by using finite number of total lags

#Applying ARMA(1,1) to price
model_price_arma11 = ARMA(train.market_value, order = (1,1))
results_price_arma11 = model_price_arma11.fit()
results_price_arma11.summary()
#ma component not SS. lets look at residuals to see what we can do differently
train['residuals_price_ARMA_1_1'] = results_price_arma11.resid

sgt.plot_acf(train.residuals_price_ARMA_1_1, zero = False, lags = 40) 
plt.title("ACF Of Residuals of Prices",size-20) 
plt.show()
#we see that 5 of the first 6 lags are significant, so lets account for these in our model by running ARMA(6,6)

model_price_arma66 = ARMA(train.market_value, order = (6,6))
results_price_arma66 = model_price_arma66.fit()
results_price_arma66.summary()

#we can address error here, by setting value for start ar lags greater than AR order of the model

model_price_arma66 = ARMA(train.market_value, order = (6,6))
results_price_arma66 = model_price_arma66.fit(start_ar_lags = 12)
results_price_arma66.summary()
#we shhould lower number of lags as 3 coeff not S.S.

#after testing, best model is ARMA(5,6) for price series
model_price_arma56 = ARMA(train.market_value, order = (5,6))
results_price_arma56 = model_price_arma56.fit(start_ar_lags = 12)
results_price_arma56.summary()

#Residual analysis:
sgt.plot_acf(train.residuals_price_ARMA_5_6, zero = False, lags = 40) 
plt.title("ACF Of Residuals of Prices",size-20) 
plt.show()
#only 3 are sig different from 0 --> residuals can be classified as white noise



###### ARMA for Returns (Stationary) vs ARMA for Price (Non-Stationary Data)
print("\n ARMA(5,1) on Stationary Data: LL = ", results_ret_arma51.llf, "\tAIC = ", results_ret_arma51.aic)
print("\n ARMA(5,6): on Non-Stationary Data: LL = ", results_price_arma56.llf, "\tAIC = ", results_price_arma56.aic)

#comparing our best ARMA models for returns and price, it becomes clear that even though we can use ARMA to model non stationary data, they perform much worse



