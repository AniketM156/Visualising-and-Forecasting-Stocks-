### Stock Market Prediction And Forecasting Using Stacked LSTM

import pandas as pd
import pandas_datareader as pdr
import numpy as np
import tensorflow as tf
tf.__version__
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

stock = 'AAPL'
key="9f99e4e180cefcfd9422cf227da2679cfffb57e8"
df = pdr.get_data_tiingo(stock, api_key=key)
df.to_csv('stock.csv')

stock_data=df.reset_index()['close']

stock_data

plt.plot(stock_data)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
stock_data=scaler.fit_transform(np.array(stock_data).reshape(-1,1))

# print(df1)

##splitting dataset into train and test split
training_size=int(len(stock_data)*0.65)
test_size=len(stock_data)-training_size
train_data,test_data=stock_data[0:training_size,:],stock_data[training_size:len(stock_data),:1]

training_size,test_size

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
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

print(X_train.shape), print(y_train.shape)

print(X_test.shape), print(ytest.shape)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

### Create the Stacked LSTM model

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(time_step,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


model.summary()

model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=10,batch_size=64,verbose=1)

### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))

### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))

### Plotting 
# shift train predictions for plotting
look_back=time_step
trainPredictPlot = numpy.empty_like(stock_data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(stock_data)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(stock_data)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(stock_data))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

len(test_data)

x_input=test_data[341:].reshape(1,-1)
x_input.shape


temp_input=list(x_input)
temp_input=temp_input[0].tolist()

#temp_input
days = 30
n_steps=100

# demonstrate prediction for next 10 days
from numpy import array

lst_output=[]
i=0
while(i<days):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

# print(lst_output)

day_new=np.arange(1,time_step +1)
day_pred=np.arange(time_step +1,time_step +1 + days)

len(stock_data)

plt.plot(day_new,scaler.inverse_transform(stock_data[1158:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))

stock_data_merged=stock_data.tolist()
stock_data_merged.extend(lst_output)
plt.plot(stock_data_merged[1200:])

stock_data_merged=scaler.inverse_transform(stock_data_merged).tolist()

plt.plot(stock_data_merged)

from datetime import datetime, timedelta

def generate_dates(start_date, n):
    dates = []
    current_date = start_date
    
    for _ in range(n):
        dates.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
    
    return dates

# Example usage
start_date = datetime(2023, 5, 31)  # Specify the start date
n = 10  # Specify the number of days

pred_date_list = generate_dates(start_date, n)
print(pred_date_list)


# original_dates = df['date'].to_list()
# original_dates


predicted_values = scaler.inverse_transform(lst_output)
# predicted_values

model.save('models/APPl_stock-100_epochs.h5')