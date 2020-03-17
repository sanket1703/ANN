import math 
import pandas_datareader as web
import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df = web.DataReader('AAPL' , data_source = 'yahoo' , start = '2012-01-01',end = '2019-12-17')
#df1 = web.DataReader('GSFC.BO' , data_source = 'yahoo' , start = '2012-01-01',end = '2020-03-14')

plt.figure(figsize = (16,8))
plt.title('Close price history')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close price ($)',fontsize=18)
plt.show()


data = df.filter(['Close'])
dataset = data.values
train_len = math.ceil(len(dataset)*0.8)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:train_len ,:] 
#Split the data into x_train and y_train 
#This is something similar to what we have done in embeddings in image processing
#after a batch of 60 we can take the 61st close price as reference

x_train =[] #training features

y_train = []  #target variable

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    if i<=61:
        print(x_train[0].shape)
        print(len(y_train))
        print(x_train)
        print(y_train)
        print()
#because it's a list and it's appending
x_train , y_train = np.array(x_train) , np.array(y_train)

x_train = np.reshape(x_train , (x_train.shape[0],x_train.shape[1],1))
x_train.shape


model= Sequential()
model.add(LSTM(50,return_sequences = True , input_shape = (x_train.shape[1],1)))
model.add(LSTM(50,return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer = 'adam',loss = 'mean_squared_error')
model.fit(x_train,y_train,batch_size = 1,epochs = 1)

test_data = scaled_data[train_len - 60:,:]

x_test = []
y_test = dataset[train_len:,:]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
x_test = np.array(x_test) 
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))   
pred = model.predict(x_test)
pred = scaler.inverse_transform(pred)
rmse = np.sqrt(np.mean( (pred-y_test)**2))

train = data[:train_len]
valid = data[train_len:]

valid['Predictions'] = pred
#Visualization 

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date' , fontsize = 18)
plt.ylabel('Close Price USD' , fontsize =18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Val','Predictions'],fontsize = 18)
plt.show()

#getting quote for a future price








