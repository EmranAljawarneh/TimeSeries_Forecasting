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