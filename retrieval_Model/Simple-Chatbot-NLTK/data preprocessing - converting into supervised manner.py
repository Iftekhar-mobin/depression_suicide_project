#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import sys
from matplotlib import pyplot
get_ipython().run_line_magic('matplotlib', 'inline')

# Path to the data txt file on disk.
data_path = 'English_for_Today.txt'


# In[8]:


# Here mecab tokenizer needed to tokenize the input data 
# Vectorize the data.
input_texts = []
target_texts = []

with open(data_path, 'r', encoding='utf-8') as f:
    #lines = f.read().split('\n')
    lines = f.read()

#print(lines)   

# Here mecab tokenizer needed to tokenize the input data 
wordsList = nltk.word_tokenize(lines)
#print(wordsList)


# In[14]:


npwordsList = np.asarray(wordsList) 

encoded = LabelEncoder()
e = encoded.fit_transform(npwordsList)

# we can also use padding technique here to make the length of number of token even number
# shaping according to desired pattern samples, Timestep, Predicted output length 
d = e[:-2].reshape(npwordsList[:len(npwordsList)-2].shape[0]//10,10)
d


# In[31]:


#Index(['Col1(C-3)', 'Col2(C-3)', 'Col3(C-3)', 'Col4(C-3)', 'Col5(C-3)',
#       'Col6(C-3)', 'Col7(C-3)', 'Col8(C-3)', 'Col9(C-3)', 'Col10(C-3)',
#       'Col1(C-2)', 'Col2(C-2)', 'Col3(C-2)', 'Col4(C-2)', 'Col5(C-2)',
#       'Col6(C-2)', 'Col7(C-2)', 'Col8(C-2)', 'Col9(C-2)', 'Col10(C-2)',
#       'Col1(C-1)', 'Col2(C-1)', 'Col3(C-1)', 'Col4(C-1)', 'Col5(C-1)',
#       'Col6(C-1)', 'Col7(C-1)', 'Col8(C-1)', 'Col9(C-1)', 'Col10(C-1)',
#       'Col1(C)', 'Col2(C)', 'Col3(C)', 'Col4(C)', 'Col5(C)', 'Col6(C)',
#       'Col7(C)', 'Col8(C)', 'Col9(C)', 'Col10(C)', 'Col1(C+1)', 'Col2(C+1)',
#       'Col3(C+1)', 'Col4(C+1)', 'Col5(C+1)', 'Col6(C+1)', 'Col7(C+1)',
#       'Col8(C+1)', 'Col9(C+1)', 'Col10(C+1)', 'Col1(C+2)', 'Col2(C+2)',
#       'Col3(C+2)', 'Col4(C+2)', 'Col5(C+2)', 'Col6(C+2)', 'Col7(C+2)',
#       'Col8(C+2)', 'Col9(C+2)', 'Col10(C+2)', 'Col1(C+3)', 'Col2(C+3)',
#       'Col3(C+3)', 'Col4(C+3)', 'Col5(C+3)', 'Col6(C+3)', 'Col7(C+3)',
#       'Col8(C+3)', 'Col9(C+3)', 'Col10(C+3)', 'Col1(C+4)', 'Col2(C+4)',
#       'Col3(C+4)', 'Col4(C+4)', 'Col5(C+4)', 'Col6(C+4)', 'Col7(C+4)',
#       'Col8(C+4)', 'Col9(C+4)', 'Col10(C+4)'],
#      dtype='object')
        
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('Col%d(C-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('Col%d(C)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('Col%d(C+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

n_hours = 3
n_features = 5
n_ahead = 10

# frame as supervised learning
reframed = series_to_supervised(d, n_hours, n_features)

st = (n_hours+1)*n_ahead + 1  

# drop columns we don't want to predict
deletedcol =    list(reframed.columns)[st           :  st+n_ahead-1] +                 list(reframed.columns)[st+n_ahead   :  st+n_ahead*2-1] +                 list(reframed.columns)[st+n_ahead*2 :  st+n_ahead*3-1] +                 list(reframed.columns)[st+n_ahead*3 : ] 
#print("deleted column",deletedcol)

reframed.drop(deletedcol, axis=1, inplace=True)
#print(reframed.head())

#reframed.to_csv(r'outNum.csv')


# In[52]:


# decoding to see the formation


# split into train and test sets
values = reframed.values
train_size = int(len(values) * 0.8)
test_size = len(values) - train_size
train, test = values[0:train_size,:], values[train_size:,:]

input_col = (n_hours+1)*n_ahead

## split into input and outputs
train_X, train_y = train[:, :input_col], train[:, -4:]
test_X, test_y = test[:, :input_col], test[:, -4:]
#print(train_X, train_X.shape, train_y, train_y.shape)


# In[53]:


print(train_X.shape, test_y.shape)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# In[57]:


# design network
model = Sequential()
model.add(LSTM(160, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(4))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=40, validation_data=(test_X, test_y), verbose=2, shuffle=False)


# In[62]:


# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# In[63]:


# make a prediction
yhat = model.predict(test_X)
yhat


# In[66]:


yhat = yhat.reshape(-1)
print(yhat.shape)
encoded.inverse_transform()


# In[48]:



#test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
# invert scaling for forecast
#inv_yhat = concatenate((yhat, test_X[:, -7:]), axis=1)
inv_yhat = scaler.inverse_transform(yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
#inv_y = concatenate((test_y, test_X[:, -7:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

