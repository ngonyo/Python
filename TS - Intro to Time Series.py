# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 11:38:12 2022

@author: Nathan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
raw_csv = pd.read_csv('/Users/nathan/Documents/Python/Datasets/Index2018.csv')
raw_csv
#create copy of dataframe
df_comp = raw_csv.copy()
df_comp.head()
df_comp.describe()
df_comp.isna() 
df_comp.isna().sum() #we can see no missing values present


##### plotting the data:
df_comp.spx.plot()
df_comp.spx.plot(figsize=(20,5), title = "S&P500 Over Time")
df_comp.ftse.plot(figsize=(20,5), title = "FTSE Over Time")
#notice if we run both plots at same time it will chart both on same plot: but we will have to specify title or it will use last one

df_comp.spx.plot(figsize=(20,5), title = "S&P500 Over Time")
df_comp.ftse.plot(figsize=(20,5), title = "FTSE Over Time")
plt.title("S&P v FTSE")
plt.show()


##### qq / norm plot:
import scipy.stats
import pylab
scipy.stats.probplot(df_comp.spx, plot = pylab)
pylab.show()


##### length of the time period:
df_comp.date.describe()
#we must convert the date column into a datetime object
#this assumed we are plugging in a string in mm/dd/yyyy form. our data is in dd/mm/yyyy
df_comp.date = pd.to_datetime(df_comp.date, dayfirst = True)
df_comp.head()
df_comp.date.describe() #this works better now


##### setting date as the index
df_comp.set_index("date", inplace = True) #inplace tells in place of integer indexes
df_comp.head() #notice how date is now on left of dataframe as opposed to integer list


##### setting desired frequency
df_comp = df_comp.asfreq('d') 
#values of this argument take letters, h-hourly, w-weekly, d-daily, m-monthly etc. HOWEVER year = annual = a
df_comp.head() #notice how weekends return NA
df_comp = df_comp.asfreq('b')  #setting to only business days
df_comp.head()

##### handling missing values
df_comp.isna().sum()
#setting the freq to business days must have generated 8 missing values
#methods for handling missing values:
#filna() method: 
#1. front filling: fills with previous period value 
#2. back filling: fills with next period value
#3. assigning same value: assigns argument(ex mean of all values) to all missing values - not ideal for TS with trend
df_comp.spx = df_comp.spx.fillna(method = "ffill") #front filling spx values
df_comp.isna().sum() #we can see this worked for spx column
df_comp.ftse = df_comp.spx.fillna(method = "ffill") #back filling ftse values
df_comp.isna().sum()
df_comp.dax = df_comp.dax.fillna(value = df_comp.dax.mean()) #mean filling dax values
df_comp.isna().sum()

##### simplifying dataset
#lets create a new column called market value and assign the sp500 series to it
df_comp['market_value'] = df_comp.spx
df_comp.describe()
#lets delete all other columns of the dataset which we wont use in our analysis
del df_comp['spx'] #one at a time
del df_comp['dax'], df_comp['ftse'], df_comp['nikkei'] #multiple at a time
df_comp.describe()


##### creation of testing and training sets
#iloc denotes index location, len denotes length
size = int(len(df_comp)*0.8)
df_train = df_comp.iloc[:size] #train goes up until 80% cutoff point
df_test = df_comp.iloc[size:] #test goes past 80% cutoff
df_train.tail() #verify
df_train.head()
df_test.head()


##### white noise analysis
#for the purposes of this analysis, white noise requires constant mean, var no autocorrelation
#lets generate white noise series, store data in new wn variable. we will create random series with mean and var from sp500 data
wn = np.random.normal(loc = df_train.market_value.mean(), scale = df_train.market_value.std(), size = len(df_train))
df_train['wn'] = wn
df_train.describe()
#lets plot the white noise series
df_train.wn.plot(title = 'white noise time series')
df_train.wn.plot(title = 'white noise time series', figsize = (20,5))
#layering graphs with same scale:
df_train.wn.plot(title = 'white noise time series', figsize = (20,5), ylim = (0,2500))
df_train.market_value.plot(title = 'white noise time series', figsize = (20,5), ylim = (0,2500))
#we can clearly see that the sp500 is not white noise

##### random walk
#random walk can be defined as TS where values persist over time and the differences between periods are white noise
#in a random walk, yt = yt-1 + et where et is white noise residual term
#our sp500 time series follows a random walk much more closely than white noise


##### stationarity
#weak form stationarity or "covariance" stationarity requires:
#constant mean, constant variance, cov(Xn, Xn+k) = cov(Xm, Xm+k)
#white noise is weakly stationary, as said before
#strict stationarity: two samples of TS of identical size have identical distribution
#strict stationarity rarely observed, usually care about weak stationarity


##### stationarity testing using dickey fuller test
#df test will be used to test stationairity
#Ho: phi<1 (ACF(1)<1) H1: (ACF(1)=1) --> stationary
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
sns.set()

#df will come from stats model tsa package
sts.adfuller(df_train.market_value) 
#first line is test statistic, then p value, then number of lags used in regression, then number of observations used in analysis
#1,5,10% critical values includes blow
#lets run on white noise series
sts.adfuller(df_train.wn) 
#we can see the white noise data is clearly stationary. sp500 is not

##### seasonality
#recall we can decompose TS using additive method, or multiplicative method
#additive decomp:
s_dec_additive = seasonal_decompose(df_train.market_value, model = "additive")
s_dec_additive.plot()
#seasonal is a block because values are oscilating on a very low time frame between 0,-.2
#therefore is no concrete cyclical pattern shown with naive decomposition
#multiplicative decomp:
s_dec_multiplicative = seasonal_decompose(df_train.market_value, model = "multiplicative")
s_dec_multiplicative.plot()

##### autocorrelation function
#statsmodels.tsaplots has built in function for plotting ACF
sgt.plot_acf(df_train.market_value, lags = 40, zero = False) #ignore ACF(0)
plt.title("ACF S&P", size = 24)
plt.show()
#now ACF of white noise series
sgt.plot_acf(df_train.wn, lags = 40, zero = False) #ignore ACF(0)
plt.title("ACF WN", size = 24)
plt.show()
#notice how all ACF coefficients for wn series fall within blue bands, are not statistically significant

##### partial autocorrelation function
#while ACF gives indirect effect of each lagged period on TS, PACF gives isolated, direct effect holding all other lags constat
sgt.plot_pacf(df_train.market_value, lags = 40, zero = False, method = ('ols'))
plt.title("PACF Sp500", size = 24)
plt.show()
#PACF for t-2 would cancel out the effect t-2 had on t-1 had on t, wheras ACF would not cancel that out
#now PACF for white noise
sgt.plot_pacf(df_train.wn, lags = 40, zero = False, method = ('ols'))
plt.title("PACF White Noise", size = 24)
plt.show()

##### exporting as csv
#lets export our dataframe as csv to use in new file for our modeling
df_train.to_csv('/Users/nathan/Documents/Python/Datasets/spx_train.csv')
df_test.to_csv('/Users/nathan/Documents/Python/Datasets/spx_test.csv')








