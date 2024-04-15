from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
Train_data = log_data.drop(labels=['DisclosedUnixTime'], axis=1)

Std_Scaler = StandardScaler()
Std_feature_transform = Std_Scaler.fit_transform(Train_data)

# Split the data
timesplit= TimeSeriesSplit(n_splits=10)
for train_index, test_index in timesplit.split(Std_feature_transform):
        X_train, X_test = Std_feature_transform[:len(train_index)], Std_feature_transform[len(train_index): (len(train_index)+len(test_index))]
        y_train, y_test = Target_data[:len(train_index)].values.ravel(), Target_data[len(train_index): (len(train_index)+len(test_index))].values.ravel()

# Build the SVR model
svr = SVR(kernel='rbf', C=100, epsilon=100.0)
svr_fit = svr.fit(X_train, y_train)
svr_prediction = svr.predict(X_test)
print(len(svr_prediction))

# Evaluate the model 
MSE = round(mean_squared_error(y_test, svr_prediction), 9)
RMSE = round(mean_squared_error(y_test, svr_prediction, squared=False), 9)
MAE = round(mean_absolute_error(y_test, svr_prediction), 9)

x = np.arange(1)
plt.bar(x-0.2, MSE, width=0.1, color='red')
plt.bar(x, RMSE, width=0.1, color='orange')
plt.bar(x+0.2, MAE, width=0.1, color='blue')

plt.xticks(x, ['MSE', 'RMSE'])
plt.xlabel("Metrics")
plt.ylabel("Scores")
plt.legend(["MSE", "RMSE", "MAE"])
plt.show()

"""# Test Stock Price"""

testcolumns = ['SecuritiesCode', 'Open', 'High', 'Low', 'Close', 'Volume']
testdata = pd.read_csv('test_stock_prices.csv', usecols=testcolumns)
testdata.shape

log_testdata = np.log(testdata)   # log() == loge()
plt.plot(log_testdata)
plt.show()

log_testdata = log_testdata.replace([np.inf, -np.inf], np.nan)
log_testdata = log_testdata.fillna(log_data.mean())

Std_Scaler = StandardScaler()
Std_feature_transform = Std_Scaler.fit_transform(log_testdata)
Std_feature_transform = pd.DataFrame(Std_feature_transform, columns=log_testdata.columns, index=log_testdata.index)

test_svr_prediction = svr.predict(Std_feature_transform)
print(len(test_svr_prediction))

print(test_svr_prediction)

test_svr_prediction = np.exp(test_svr_prediction)
print(test_svr_prediction)

x = 0
for i in test_svr_prediction:
  print(pd.Timestamp(i, unit='s'))
  x=x+1
print(x)
