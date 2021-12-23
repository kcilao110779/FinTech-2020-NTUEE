# -*- coding: utf-8 -*-
"""HW4

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1r4-kOtcPGFfoU1i_O9-T9CSRfJNvJq33
"""

# !pip install mplfinance
import pandas as pd
import mplfinance as mpf
import datetime
from datetime import date
import numpy as np
from sklearn import preprocessing

# url = 'https://launchpad.net/~mario-mariomedina/+archive/ubuntu/talib/+files'
# !wget $url/libta-lib0_0.4.0-oneiric1_amd64.deb -qO libta.deb
# !wget $url/ta-lib0-dev_0.4.0-oneiric1_amd64.deb -qO ta.deb
# !dpkg -i libta.deb ta.deb
# !pip install ta-lib

import talib
from talib import  abstract

# data = pd.read_csv("S_P.csv",parse_dates=True)

data = pd.read_csv("S_P.csv",parse_dates=True)
start_date = (data['Date']== '2019-01-02' )

idx = np.where(start_date)[0][0]


data = pd.read_csv("S_P.csv",parse_dates=True, index_col='Date')

data.columns = map(str.lower, data.columns)
KD = abstract.STOCH(data)
data['K'] = KD['slowk']
data['D'] = KD['slowd']
data = data.fillna(0)

data_2019 = data.iloc[idx:,:]

# mpf.plot(data_2019, type = 'candle',figratio=(10,2),figscale=0.5, mav=(10,30), volume=True, savefig='candle&ma&volume.png')

mpf.plot(data_2019, type = 'candle',figratio=(7,2),figscale=0.5, mav=(10,30), volume=True,savefig='candle&ma&volume.png')
data_2019['K'].plot(figsize=(24,8))
data_2019['D'].plot(figsize=(24,8))
plt.savefig("KDline.png")

sma_5 = abstract.SMA(data,10)  ## 5就是timeperiod
sma_30 = abstract.SMA(data,30)
data['sma_5'] = sma_5
data['sma_30'] = sma_30
data = data.fillna(0)



data_normalize = (data-data.min())/(data.max()-data.min())
data_normalize.pop('adj close')

data_2018 = (data_normalize.index.year == 2018)
idx = np.where(data_2018)[0][0]
trainData = data_normalize.iloc[:idx,:]
testData = data_normalize.iloc[idx:, :]

y_trainData = trainData.pop('close')
y_testData = testData.pop('close')

x_train = []   #預測點的前 30 天的資料
y_train = []   #預測點
x_test = []
y_test = []

for i in range(30, trainData.shape[0]):  # data.shape[0] 是訓練集總數
    x_train.append(trainData.iloc[i-30:i, :])
    y_train.append(y_trainData.iloc[i])
for i in range(30, testData.shape[0]):
    x_test.append(testData.iloc[i-30:i, :])
    y_test.append(y_testData.iloc[i]
                  )
x_train, y_train = np.array(x_train), np.array(y_train)  # 轉成numpy array的格式，以利輸入 RNN
x_test, y_test = np.array(x_test), np.array(y_test)

import tensorflow as tf
import keras
from keras.layers import LSTM
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Dropout, Flatten, SimpleRNN, GRU
from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dropout

def create_RNN(x_train, y_train, x_test, y_test):
  batch_size = None
  steps = 30
  input_dim = 8
  epochs = 20
  model = Sequential()
  # 加 RNN 隱藏層(hidden layer)
  model.add(SimpleRNN(
      # 如果後端使用tensorflow，batch_input_shape 的 batch_size 需設為 None.
      # 否則執行 model.evaluate() 會有錯誤產生.
      batch_input_shape=(batch_size, steps, input_dim), 
      units= 50,
      unroll=True,
  ))
  model.add(Dropout(0.2))
  model.add(Dense(10,activation='relu'))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
  hist = model.fit(x_train, y_train,
            batch_size=batch_size, epochs=epochs,
            verbose=1, validation_data=(x_test, y_test))
  return hist, model

def create_LSTM(x_train, y_train, x_test, y_test):
  batch_size = None
  steps = 30
  input_dim = 8
  epochs = 20
  lstm = Sequential()
  # 加 RNN 隱藏層(hidden layer)
  lstm.add(LSTM(
      # 如果後端使用tensorflow，batch_input_shape 的 batch_size 需設為 None.
      # 否則執行 model.evaluate() 會有錯誤產生.
      batch_input_shape=(batch_size, steps, input_dim), 
      units= 50,
      unroll=True,
  )) 
  lstm.add(Dense(10,activation='relu'))
  lstm.add(Dense(1))
  lstm.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
  hist = lstm.fit(x_train, y_train,
            batch_size=batch_size, epochs=epochs,
            verbose=1, validation_data=(x_test, y_test))
  print(lstm.summary())
  return hist, lstm

def create_GRU(x_train, y_train, x_test, y_test):
  batch_size = None
  steps = 30
  input_dim = 8
  epochs = 20
  model = Sequential()
  # 加 RNN 隱藏層(hidden layer)
  model.add(GRU(
      # 如果後端使用tensorflow，batch_input_shape 的 batch_size 需設為 None.
      # 否則執行 model.evaluate() 會有錯誤產生.
      batch_input_shape=(batch_size, steps, input_dim), 
      units= 50,
      unroll=True,
  )) 
  model.add(Dense(10,activation='relu'))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
  hist = model.fit(x_train, y_train,
            batch_size=batch_size, epochs=epochs,
            verbose=1, validation_data=(x_test, y_test))
  return hist, model

LSTM_hist, LSTM = create_LSTM(x_train,y_train,x_test,y_test)

RNN_hist, RNN = create_RNN(x_train,y_train,x_test,y_test)

GRU_hist, GRU = create_GRU(x_train,y_train,x_test,y_test)

import matplotlib.pyplot as plt

def plot_loss(hist, modelname):
  name = modelname+" loss"
  plt.plot(hist.history['loss'])
  plt.plot(hist.history['val_loss'])
  plt.legend(['Train', 'Test'], loc='upper right')
  plt.title(modelname+" loss")
  plt.savefig(name)
  plt.show()

def denormalized(data, data_normalize):
  denormal = data_normalize*(data['close'].max()-data['close'].min())+data['close'].min()
  return denormal
def plot_predict_close(x_test, y_test, model, modelname):
  predict_y = model.predict(x_test)
  denormal_predict_y = denormalized(data, predict_y)
  denormal_y = denormalized(data, y_test)
  name = modelname+' prediction'
  plt.plot(denormal_y)
  plt.plot(denormal_predict_y)
  plt.legend(['True', 'Predict'], loc='upper right')
  plt.title(name)
  plt.savefig(name)
  plt.show()

plot_loss(RNN_hist, 'RNN')
plot_predict_close(x_test, y_test, RNN, "RNN")
plot_loss(LSTM_hist, 'LSTM')
plot_predict_close(x_test, y_test, LSTM, "LSTM")
plot_loss(GRU_hist, 'GRU')
plot_predict_close(x_test, y_test, GRU, "GRU")

data_2020 = pd.read_csv("2020.csv",parse_dates=True, index_col='Date')
data_2020.columns = map(str.lower, data_2020.columns)
KD = abstract.STOCH(data_2020)
data_2020['K'] = KD['slowk']
data_2020['D'] = KD['slowd']

sma_5 = abstract.SMA(data_2020,10)  ## 5就是timeperiod
sma_30 = abstract.SMA(data_2020,30)
data_2020['sma_5'] = sma_5
data_2020['sma_30'] = sma_30

data_2020 = (data_2020-data.min()) / (data.max()-data.min())
data_2020.pop('adj close')
y_testData_2020 = data_2020.pop('close')
data_2020 = data_2020.fillna(0)

x_test_2020 = []
y_test_2020 = []

for i in range(30, data_2020.shape[0]):
    x_test_2020.append(data_2020.iloc[i-30:i, :])
    y_test_2020.append(y_testData_2020.iloc[i]
                  )

x_test_2020, y_test_2020 = np.array(x_test_2020), np.array(y_test_2020)

plot_predict_close(x_test_2020, y_test_2020, RNN, "RNN_2020")
plot_predict_close(x_test_2020, y_test_2020, LSTM, "LSTM_2020")
plot_predict_close(x_test_2020, y_test_2020, GRU, "GRU_2020")
