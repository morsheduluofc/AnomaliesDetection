# Implementation of timeseries forecasting algorithm
#      SARIMAX (Seasonal Auto-Regressive Integrated Moving Average with eXogenous factors)

#import all packages
from cProfile import label
from xmlrpc.client import DateTime
import pandas as pd
from psycopg2 import Date
import numpy as np
from datetime import datetime

import math
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from sklearn.model_selection import train_test_split

from matplotlib import pyplot
import matplotlib.pyplot as plt


class SARIMAX:
    def __init__(self, path):
        #read datafile
        self.df=pd.read_csv(path)
        #print(self.df)
    
    #seperate and process datetime and target data 
    def process_data(self, datetime_cname, target_name):
        self.df=self.df[[datetime_cname, target_name]]
        self.df[datetime_cname] = pd.to_datetime(self.df[datetime_cname])
        return self.df
    
    #split the target in tran and test set. Log the data
    def train_test_data(self, df, cname):
        actual_vals=df[cname].values
        x_train, x_test= train_test_split(actual_vals, test_size=0.30, shuffle=False)
        x_train_log, x_test_log=np.log10(x_train),np.log10(x_test)
        return x_train, x_train_log, x_test, x_test_log

    #train and predict the forecasting model: SARIMAX(Seasonal Auto-Regressive Integrated Moving Average with eXogenous factors)
    def train_forecasting_model(self, df, train, test, datetime_cname, total_testdata):
        history=[x for x in train]
        predictions=list()
        predict_log=list()
        
        my_order=(1,1,1)
        my_seasonal_order=(0,1,1,7)

        for t in range(len(test)):
            model=sm.tsa.SARIMAX(history, order=my_order, seasonal_order=my_seasonal_order,enforce_stationarity=False,enforce_invertibility=False)
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            predict_log.append(output[0])
            yhat = 10**output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            print('predicted=%f, expected=%f' % (output[0], obs))
        #error=math.sqrt(mean_absolute_error(test_log, predict_log))
        #print('Test rmse: %.3f' % error)

        predicted_df=pd.DataFrame()
        predicted_df['date']=df[datetime_cname][-total_testdata:]
        predicted_df['actuals']=10**test
        predicted_df['predicted']=predictions
        
        return predicted_df

    #draw the prediction
    def draw_graph(self, df, Actual='actuals',Predicted='predicted'):
        plt.figure(figsize=(12,7))
        pyplot.plot(df[Actual],label='Actual')
        pyplot.plot(df[Predicted],color='red',label='Predicted')
        pyplot.legend(loc='upper right')
        pyplot.show()