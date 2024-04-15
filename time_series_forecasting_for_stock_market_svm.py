import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

columns = ['SecuritiesCode', 'Open', 'High', 'Low', 'Close', 'Volume', 'DisclosedUnixTime']
data = pd.read_csv('Stock_Market.csv', usecols=columns)
data.shape

# print(data.isna().any())

# plt.plot(data)
# plt.show()

log_data = np.log(data)   # log() == loge()
# plt.plot(log_data)
# plt.show()

log_data = log_data.replace([np.inf, -np.inf], np.nan)
log_data = log_data.fillna(log_data.mean())

Target_data = log_data['DisclosedUnixTime']
Train_data = log_data.drop(labels=['DisclosedUnixTime'], axis=1)

Std_Scaler = StandardScaler()
Std_feature_transform = Std_Scaler.fit_transform(Train_data)
# Std_feature_transform = pd.DataFrame(Std_feature_transform, columns=Train_data.columns, index=Train_data.index)

from sklearn.model_selection import TimeSeriesSplit
timesplit= TimeSeriesSplit(n_splits=10)
for train_index, test_index in timesplit.split(Std_feature_transform):
        X_train, X_test = Std_feature_transform[:len(train_index)], Std_feature_transform[len(train_index): (len(train_index)+len(test_index))]
        y_train, y_test = Target_data[:len(train_index)].values.ravel(), Target_data[len(train_index): (len(train_index)+len(test_index))].values.ravel()

from sklearn.svm import SVR
svr = SVR(kernel='rbf', C=100, epsilon=100.0)

svr_fit = svr.fit(X_train, y_train)

svr_prediction = svr.predict(X_test)
print(len(svr_prediction))

from sklearn.metrics import mean_squared_error, mean_absolute_error
print('MSE: ', mean_squared_error(y_test, svr_prediction))
print('RMSE: ', mean_squared_error(y_test, svr_prediction, squared=False))
print('MAE', mean_absolute_error(y_test, svr_prediction))

print(svr_prediction)

exp_svr_prediction = np.exp(svr_prediction)
print(exp_svr_prediction)

'''x = 0
for i in exp_svr_prediction:
  print(pd.Timestamp(i, unit='s'))
  x+=1
print(x)'''

"""# Test Stock Price"""

'''testcolumns = ['SecuritiesCode', 'Open', 'High', 'Low', 'Close', 'Volume']
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
print(x)'''
