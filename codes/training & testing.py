#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, InputLayer, TimeDistributed, Bidirectional
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:
train_X = np.concatenate([np.load('M10_train_X.npy'), np.load('M10_train_X2.npy')], axis=0)
train_y = np.concatenate([np.load('M10_train_y.npy'), np.load('M10_train_y2.npy')], axis=0)
print(train_X.shape)
print(train_y.shape)


# In[3]:
val_X = np.load('M10_val_X.npy')
val_y = np.load('M10_val_y.npy')
print(val_X.shape)
print(val_y.shape)


# In[4]:
test_X = np.load('M10_test_X.npy')
test_y = np.load('M10_test_y.npy')
print(test_X.shape)
print(test_y.shape)


# In[5]:
n_timesteps = 10

model = Sequential()

model.add(Dense(128, input_shape=(n_timesteps, 513), activation='relu'))

model.add(Bidirectional(LSTM(128, return_sequences=True, activation='relu')))

model.add(LSTM(256, return_sequences=True, activation='relu'))

model.add(TimeDistributed(Dense(2, activation='relu')))

model.compile(loss='mse' , optimizer= 'adam' , metrics=['mse'])
print(model.summary())


# In[6]:


history = model.fit(train_X, train_y, epochs=500, batch_size=128, verbose=2, validation_data=(val_X, val_y))


# In[7]:
print(history.history.keys())


# In[8]:
plt.figure(figsize=(8, 5))
plt.title('Loss', fontsize=14)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.show()


# In[9]:
model.save('ann_lstm_lstm_bs128_ep500.h5')


# In[10]:
y_pred = model.predict(test_X)


# In[13]:
sbp_mse = mean_squared_error(test_y[:, :, 0], y_pred[:, :, 0])
sbp_rms = sqrt(mean_squared_error(test_y[:, :, 0], y_pred[:, :, 0]))
sbp_mae = mean_absolute_error(test_y[:, :, 0], y_pred[:, :, 0])
print('SBP MSE:\t', sbp_mse)
print('SBP RMSE:\t', sbp_rms)
print('SBP MAE:\t', sbp_mae)

dbp_mse = mean_squared_error(test_y[:, :, 1], y_pred[:, :, 1])
dbp_rms = sqrt(mean_squared_error(test_y[:, :, 1], y_pred[:, :, 1]))
dbp_mae = mean_absolute_error(test_y[:, :, 1], y_pred[:, :, 1])
print('DBP MSE:\t', dbp_mse)
print('DBP RMSE:\t', dbp_rms)
print('DBP MAE:\t', dbp_mae)

