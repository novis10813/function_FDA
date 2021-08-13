import pandas as pd
import numpy as np
import math
import mplfinance as mpf
import talib

### use 'frofunctionm ipynb.fs.full.function_FDA import *function' to use these
### df is pandas DataFrame

#caculating daily return
def daily_return(df):
    return df.pct_change(1).dropna()

# 每個值取log後相減
def logrx(df):
    return np.log(df).diff().dropna()

# caculating skewness and kurtosis
def my_skewness(df):
    T = len(df)
    y = df-df.mean()
    return T*math.sqrt(T-1)/(T-2)*((y**3).sum())/(((y**2).sum())**(3/2))

def my_kurtosis(df):
    T = len(df)
    y = df-df.mean()
    
    f1 = T*(T+1)*(T-1)/((T-2)*(T-3))
    f2 = 3*((T-1)**2)/((T-2)*(T-3))
    
    return f1*((y**4).sum())/(((y**2).sum())**2)**2-f2

# a function for calculating 1/N portfolio return
def portfolio_mean_daily_return(df):
    return df.pct_change(1).dropna().iloc[:, :].mean(axis=1)

# a function for calculating BH portfolio return
def portfolio_bh_daily_return(df):
    x = df.pct_change(1).dropna() + 1
    x = x.cumprod().iloc[:, :].mean(axis=1)  # get the mean of cumulative product
    
    return x.pct_change(1).dropna()

## functions for calculating volatilities
# Rolling Parkinson volatility
def Parkinson_volatility(df, t): #high price and low price are required in tha dataframe, t = period
    x = ((np.log(df['High']) - np.log(df['Low']))**2).rolling(t).sum().reindex(df.index)
    x = x/t*math.log(2)
    return np.sqrt(x.dropna())

# Garman-Klass (GK) volatility:
def Garman_klass(df, t):  # requires open, low, high, close price in the dataframe
    x = ((np.log(df['High']) - np.log(df['Low']))**2).rolling(t).mean().reindex(df.index)
    y = ((np.log(df['Close']) - np.log(df['Open']))**2).rolling(t).mean()
    z = 0.5*x-(2*math.log(2)-1)*y
    return np.sqrt(z.dropna())

# Roger, Satchell and Yoon (RSY) volatility
def RSY(df):
    x = (np.log(df['High']) - np.log(df['Close']))*(np.log(df['High']) - np.log(df['Open']))
    y = (np.log(df['Low']) - np.log(df['Close']))*(np.log(df['Low']) - np.log(df['Open']))
    z = x+y
    return np.sqrt(z)

#Yang and Zhang (YZ) volatility
def YZ(df, alpha=1.34):
    k = (alpha - 1)/(alpha + len(df))

# Draw candlesticks
def draw_candlestick(df):
    return mpf.plot(df,type='candle', volume=True)
    