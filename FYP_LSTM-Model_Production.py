# import os ## comment on production 
import pandas as pd
import pandas_datareader as pdr
import yfinance as yf
import numpy as np
import tensorflow as tf
# tf.__version__ #comment on production
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import math
from sklearn.metrics import mean_squared_error

stock = 'RELIANCE.NS'  # 1
# stock = 'HDFCBANK.NS'  # 2
# stock = 'M&M.NS'  # 3
# stock = 'TATAMOTORS.NS'  # 4
# stock = 'BAJFINANCE.NS'  # 5
# stock = 'INFY.NS'  # 6
# stock = 'ICICIBANK.NS'  # 7
# stock = 'TCS.NS'  # 8
# stock = 'WIPRO.NS'  # 9
# stock = 'HINDUNILVR.NS'  # 10

time_step = 50
epoch_value = 100
days = 30

# stock_ticker = yf.Ticker(stock) # from yfianance  #comment on production
# stock_data = stock_ticker.history(period="max")  #comment on production
# stock_data = stock_data.loc['2018-5-31':]  #comment on production

# -------------------------Creating Date CSV-------------------------------------->
# This code is writtern to get date in csv forat variable dates takes stock_data 
# then deletes all coloumns of it and then converts only remaining row which is dates to csv
# dates = stock_data.copy()  #comment on production
# del dates['Open'],dates['Close'],dates['Low'],dates['High'],dates['Volume'],dates['Dividends'],dates['Stock Splits']  #comment on production
# dates.to_csv('/content/universal_dates.csv')  # Used t create a universal dates csv   #comment on production
# dates  #comment on production
# -------------------------------------------------------------------------------->

# -------------------------Creating Stock Data CSV-------------------------------->
# df = stock_data  #comment on production
# df.to_csv('/content/{}_dataset.csv'.format(stock), index=False)  #comment on production
# stock_data = pd.read_csv('/content/{}_dataset.csv'.format(stock))  #comment on production
# stock_data  #comment on production

# -------------------------Merging Stock data and Dates CSV----------------------->
# Read the CSV files
# df_stock_values = pd.read_csv('/content/{}_dataset.csv'.format(stock))  #comment on production
# df_dates = pd.read_csv('/content/universal_dates.csv')  #comment on production

# Merge the DataFrames based on a common column  ##
# stock_data = pd.merge(df_stock_values,df_dates, left_on=None)
# stock_data = pd.concat([df_stock_values, df_dates], axis=1)  #comment on production
# os.remove('/content/{}_dataset.csv'.format(stock))  #comment on production

# -------------------------Finalising merged file--------------------------------->
# Save the merged DataFrame to a new CSV file
# stock_data.to_csv('/content/merged_{}_dataset.csv'.format(stock), index=False)  #comment on production
# stock_data  #comment on production

stock_data = pd.read_csv('/content/merged_{}_dataset.csv'.format(stock)) # put adress where csv files are stored

close = stock_data['Close']
open = stock_data['Open']
high = stock_data['High']
low = stock_data['Low']
# close#comment on production
# open#comment on production
# high#comment on production
# low#comment on production

# df.to_csv("{}dataset.csv".format(stock),index = False)
# df.to_csv('/content/{}_dataset.csv'.format(stock), index=False)
# stock_data = pd.read_csv('/content/{}_dataset.csv'.format(stock))
# stock_data

# stock_data=df.reset_index()['close'] this is for tingo
# df
# stock_data
# stock_data=df.reset_index()['Close']
stock_data=stock_data.reset_index()['Close']
# stock_data #important

import matplotlib.pyplot as plt
# plt.plot(stock_data) #imp

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
stock_data=scaler.fit_transform(np.array(stock_data).reshape(-1,1))

# print(df1)

##splitting dataset into train and test split
training_size=int(len(stock_data)*0.65)
test_size=len(stock_data)-training_size
train_data,test_data=stock_data[0:training_size,:],stock_data[training_size:len(stock_data),:1]

# training_size,test_size #important

#train_data

import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)

# reshape into X=t,t+1,t+2,t+3 and Y=t+4
# time_step = 200
# time_step = int(input("Enter time step genreally 100 -200 :  "))
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

# print(X_train.shape), print(y_train.shape)#important

# print(X_test.shape), print(ytest.shape) #important

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

### Create the Stacked LSTM
#50 and n_steps 
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(time_step,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')

model.summary()

# epoch_value = int(input("Enter no. of epochs generally 10-50 "))
# epoch_value = 30
model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=epoch_value,batch_size=64,verbose=1)

### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

### Calculate RMSE performance metrics
# math.sqrt(mean_squared_error(y_train,train_predict))#comment on production

### Test Data RMSE
# math.sqrt(mean_squared_error(ytest,test_predict))#comment on production

### Plotting 
# shift train predictions for plotting
look_back=time_step
trainPredictPlot = numpy.empty_like(stock_data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
#using  shift test predictions for plotting
testPredictPlot = numpy.empty_like(stock_data)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(stock_data)-1, :] = test_predict
# plot baseline and predictions for range 
plt.plot(scaler.inverse_transform(stock_data))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
# plt.show()#comment on production GRAPH

l = len(test_data) #important
# days = 10
n_steps=time_step

x_input=test_data[l-n_steps:].reshape(1,-1)
# x_input.shape#comment on production 

temp_input=list(x_input)
temp_input=temp_input[0].tolist()
#temp_input

# demonstrate prediction for next 10 days
from numpy import array

lst_output=[]
i=0
while(i<days):
    
    if(len(temp_input)>n_steps):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        # print("{} day input {}".format(i,x_input))#important
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        # print("{} day output {}".format(i,yhat))#important
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        # print(yhat[0])#important
        temp_input.extend(yhat[0].tolist())
        # print(len(temp_input))#important
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)

day_new=np.arange(1,time_step +1)
day_pred=np.arange(time_step +1,time_step +1 + days)

l2 = len(stock_data)

# plt.plot(day_new,scaler.inverse_transform(stock_data[l2-n_steps:]))#comment on production
# plt.plot(day_pred,scaler.inverse_transform(lst_output))#comment on production

stock_data_merged=stock_data.tolist()
stock_data_merged.extend(lst_output)
# plt.plot(stock_data_merged[1200:])#comment on production#comment on production

stock_data_merged=scaler.inverse_transform(stock_data_merged).tolist()

# plt.plot(stock_data_merged) #comment on production

predicted_values = scaler.inverse_transform(lst_output)
#OUTPUT values ----------------->
# close
# open
# high
# low
# predicted_values

# # C:\Users\anike\My Drive\College\FYP_resources_Gdrive\Test_Codes_do_not_touch\version_controll
# from google.colab import files
# uploaded = files.upload()

prediction_dates = pd.read_csv('/content/prediction_dates.csv')  #comment on production
prediction_dates   #comment on production
predicted_values_df = pd.DataFrame(predicted_values, columns = ['Estimed Price'])   #comment on production
predicted_values_df   #comment on production
merged_stock_predicition = pd.concat([predicted_values_df, prediction_dates], axis=1)  #comment on production
merged_stock_predicition.to_csv('/content/{}_stock_predicition.csv'.format(stock))   #comment on production

