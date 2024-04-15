from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""# Reading the required files"""

columns = ['SecuritiesCode', 'Open', 'High', 'Low', 'Close', 'Volume', 'DisclosedUnixTime']
data = pd.read_csv('Stock_Market.csv', usecols=columns)

data.shape

# Non-Stationary TimeSeries test using visualization
aaa=['Open', 'High', 'Low', 'Close', 'Volume']
plt.plot(data[aaa])
plt.xlabel('\n Number of observations')
plt.ylabel('\n Feature valules')
plt.show()

# Stationary Test using Summary Statistics
X = data.values
split = round(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))

# Stationary TimeSeries test using visualization
aaa=['Open', 'High', 'Low', 'Close']
log_data = np.log(data)   # log() == loge()
plt.plot(log_data['Open'], 'b')
plt.xlabel('\n Number of observations')
plt.ylabel('\n Feature valules')
plt.show()

log_data = log_data.replace([np.inf, -np.inf], np.nan)
log_data = log_data.fillna(log_data.mean())

# Stationary Test using Summary Statistics
X = log_data.values
X = np.log(X)
split = round(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))
Target_data = log_data['DisclosedUnixTime']
Train_data = log_data.drop(labels=['DisclosedUnixTime', 'SecuritiesCode'], axis=1)

Target_data.head()

MinMax_Scaler = MinMaxScaler()
MinMax_feature_transform = MinMax_Scaler.fit_transform(Train_data, range(0, 1))
#MinMax_feature_transform = pd.DataFrame(MinMax_feature_transform, columns=Train_data.columns, index=Train_data.index)

timesplit= TimeSeriesSplit(n_splits=10)
for train_index, test_index in timesplit.split(MinMax_feature_transform):
        X_train, X_test = MinMax_feature_transform[:len(train_index)], MinMax_feature_transform[len(train_index): (len(train_index)+len(test_index))]
        y_train, y_test = Target_data[:len(train_index)].values.ravel(), Target_data[len(train_index): (len(train_index)+len(test_index))].values.ravel()

"""# train_test_split method"""

X_train, X_test, y_train, y_test = train_test_split(Train_data, Target_data, test_size=0.3, random_state=0)

print(y_train.shape, X_train.shape, X_test.shape, y_test.shape)

"""# LSTM"""

# fix random seed for reproducibility
tf.random.set_seed(7)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#Process the data for LSTM
trainX =np.array(X_train)
testX =np.array(X_test)
X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])

trainY =np.array(y_train)

#Building the LSTM Model
model = Sequential()
# 45, 55, 123, 132
model.add(LSTM(units=50, input_shape=(1, trainX.shape[1]), activation='relu', return_sequences=False))
#model.add(LSTM(units=50, dropout=0.2, input_shape=(1, trainX.shape[1]), activation='relu', return_sequences=False))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adamax')

history = model.fit(X_train, y_train, epochs=100, batch_size=1)

lstm_prediction = model.predict(X_test)
print(len(lstm_prediction))

MSE = round(mean_squared_error(y_test, lstm_prediction), 9)
RMSE = round(mean_squared_error(y_test, lstm_prediction, squared=False), 9)
MAE = round(mean_absolute_error(y_test, lstm_prediction), 9)

print('MSE: ', MSE)
print('RMSE: ', RMSE)
print('MAE', MAE)

x = np.arange(1)
plt.bar(x-0.2, MSE, width=0.1, color='red')
plt.bar(x, RMSE, width=0.1, color='orange')
plt.bar(x+0.2, MAE, width=0.1, color='blue')

plt.xticks(x, ['MSE', 'RMSE'])
plt.xlabel("Metrics")
plt.ylabel("Scores")
plt.legend(["MSE", "RMSE", "MAE"])
plt.show()

"""# Convert the Time stamp into Time Format"""

print(lstm_prediction)

exp_lstm_prediction = np.exp(lstm_prediction)
print(exp_lstm_prediction)

exp_y_test = np.exp(y_test)
print(exp_y_test)

for i in exp_lstm_prediction:
 print(pd.to_datetime(i, unit='s'))

"""# Test Stock Market"""

testcolumns = ['Open', 'High', 'Low', 'Close', 'Volume']
testdata = pd.read_csv('test_stock_prices.csv', usecols=testcolumns)
testdata.shape

log_testdata = np.log(testdata)   # log() == loge()
plt.plot(log_testdata)
plt.show()

print(log_testdata.columns)

log_testdata = log_testdata.replace([np.inf, -np.inf], np.nan)
log_testdata = log_testdata.fillna(log_data.mean())

MinMax_Scaler = MinMaxScaler()
MinMax_feature_transform = MinMax_Scaler.fit_transform(log_testdata, range(0, 1))
#MinMax_feature_transform = pd.DataFrame(MinMax_feature_transform, columns=log_testdata.columns, index=log_testdata.index)

test_data = MinMax_feature_transform.reshape(log_testdata.shape[0], 1, log_testdata.shape[1])
print(test_data.shape)

test_lstm_prediction = model.predict(test_data)
print(len(test_lstm_prediction))

print(test_lstm_prediction)

test_lstm_prediction = np.exp(test_lstm_prediction)
print(test_lstm_prediction)

x = 0
for i in test_lstm_prediction:
  print(pd.to_datetime(i, unit='s'))
  x=x+1
print(x)
