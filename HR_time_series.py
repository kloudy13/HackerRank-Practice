#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 13:54:36 2018

@author: klaudia
"""
sth = [[1339.0], [1462.0], [1702.0], [1656.0], [1439.0], [1208.0], [1613.0], [1935.0], [1964.0], [2003.0], [2023.0], [1559.0], [1274.0], [1805.0], [2051.0], [2024.0], [2049.0], [1998.0], [1441.0], [1257.0], [1559.0], [1643.0], [1464.0], [1521.0], [1576.0], [1381.0], [1372.0], [1610.0], [1926.0], [2018.0], [1930.0], [1868.0], [1551.0], [1471.0], [1954.0], [2079.0], [2061.0], [2001.0], [2031.0], [1574.0], [1165.0], [1366.0], [1384.0], [1342.0], [1543.0], [1509.0], [1598.0], [1349.0], [1625.0], [1932.0], [2079.0], [1857.0], [1443.0], [1332.0], [1149.0], [1471.0], [1595.0], [1498.0], [1433.0], [1401.0], [1178.0], [972.0], [1283.0], [1468.0], [1456.0], [1466.0], [1378.0], [1208.0], [1038.0], [1344.0], [1379.0], [1418.0], [1466.0], [1414.0], [1129.0], [948.0], [1285.0], [1373.0], [1379.0], [599.0], [614.0], [850.0], [672.0], [747.0], [732.0], [834.0], [996.0], [900.0], [792.0], [688.0], [781.0], [696.0], [834.0], [1026.0], [974.0], [1001.0], [997.0], [1110.0], [1212.0], [1301.0], [1322.0], [1253.0], [935.0], [857.0], [1082.0], [1112.0], [1291.0], [1391.0], [1384.0], [1089.0], [963.0], [1174.0], [1420.0], [1349.0], [1338.0], [1335.0], [1075.0], [952.0], [1376.0], [1586.0], [1571.0], [1543.0], [1525.0], [1223.0], [1066.0], [1555.0], [1704.0], [1745.0], [1749.0], [1687.0], [1288.0], [1152.0], [1492.0], [1728.0], [1742.0], [1732.0], [1510.0], [1288.0], [1280.0], [1659.0], [1852.0], [1823.0], [1723.0], [1416.0], [1187.0], [1014.0], [1324.0], [1618.0], [1736.0], [1552.0], [1598.0], [1158.0], [1083.0], [1383.0], [1595.0], [1540.0], [1551.0], [1447.0], [1128.0], [1057.0], [1371.0], [1746.0], [1653.0], [1726.0], [1759.0], [1297.0], [1165.0], [1480.0], [1693.0], [1744.0], [1661.0], [1575.0], [1199.0], [1062.0], [1395.0], [1555.0], [1441.0], [1399.0], [1381.0], [1287.0], [1151.0], [1394.0], [1660.0], [1761.0], [1874.0], [1863.0], [1544.0], [1340.0], [1707.0], [1983.0], [1785.0], [1725.0], [1765.0], [1520.0], [1274.0], [1763.0], [1793.0], [1853.0], [1861.0], [1733.0], [1575.0], [1304.0], [1793.0], [1886.0], [1832.0], [1993.0], [1805.0], [1521.0], [1332.0], [1813.0], [1833.0], [1633.0], [1682.0], [1699.0], [1392.0], [1249.0], [1601.0], [1827.0], [1755.0], [1560.0], [1181.0], [1039.0], [920.0], [1106.0], [1269.0], [1159.0], [1192.0], [1203.0], [1876.0], [1065.0], [1294.0], [1249.0], [1145.0], [1098.0], [1171.0], [996.0], [807.0], [1081.0], [1258.0], [1201.0], [1273.0], [1165.0], [944.0], [797.0], [948.0], [1160.0], [1387.0], [1364.0], [1061.0], [916.0], [901.0], [1047.0], [1107.0], [1234.0], [1290.0], [1027.0], [877.0], [629.0], [789.0], [977.0], [953.0], [983.0], [927.0], [870.0], [724.0], [824.0], [1015.0], [1062.0], [962.0], [1046.0], [766.0], [764.0], [876.0], [975.0], [917.0], [914.0], [1239.0], [1000.0], [753.0], [892.0], [1000.0], [1509.0], [1183.0], [955.0], [862.0], [659.0], [780.0], [916.0], [1010.0], [988.0], [1032.0], [856.0], [729.0], [917.0], [1038.0], [1146.0], [1228.0], [1210.0], [939.0], [913.0], [1045.0], [1131.0], [1076.0], [1068.0], [1073.0], [919.0], [809.0], [963.0], [1095.0], [1150.0], [950.0], [909.0], [889.0], [840.0], [1156.0], [1221.0], [1247.0], [1146.0], [1142.0], [969.0], [912.0], [1103.0], [1146.0], [1118.0], [1193.0], [1200.0], [1065.0], [908.0], [1491.0], [1968.0], [2100.0], [2402.0], [2489.0], [2110.0], [2046.0], [2288.0], [2835.0], [2620.0], [2467.0], [2262.0], [2046.0], [1711.0], [1960.0], [2241.0], [2383.0], [2463.0], [2289.0], [1847.0], [1654.0], [2040.0], [2473.0], [2492.0], [2501.0], [2566.0], [2284.0], [2063.0], [2655.0], [2878.0], [2957.0], [2710.0], [2526.0], [2105.0], [1782.0], [2242.0], [2520.0], [2246.0], [2378.0], [2545.0], [2154.0], [2022.0], [2711.0], [3142.0], [2952.0], [2882.0], [2766.0], [2104.0], [1745.0], [2043.0], [2495.0], [2555.0], [2572.0], [2723.0], [2276.0], [1936.0], [2472.0], [2800.0], [2843.0], [2670.0], [2769.0], [2267.0], [2213.0], [2695.0], [2801.0], [2608.0], [2418.0], [1940.0], [1610.0], [1459.0], [1972.0], [2292.0], [2573.0], [2630.0], [2448.0], [2022.0], [1869.0], [2313.0], [2551.0], [2653.0], [2595.0], [2373.0], [2088.0], [1954.0], [2374.0], [2780.0], [2674.0], [2675.0], [2445.0], [2095.0], [1806.0], [2279.0], [2639.0], [2616.0], [2152.0], [2055.0], [1775.0], [1682.0], [2151.0], [2386.0], [2519.0], [2381.0], [2394.0], [1832.0], [1729.0], [1953.0], [2224.0], [2164.0], [2121.0], [2103.0], [1693.0], [1557.0], [1859.0], [1884.0], [1949.0], [1861.0], [1727.0], [1489.0], [1197.0], [1385.0], [1412.0], [1165.0], [957.0], [475.0], [1301.0], [1148.0], [1272.0], [1333.0], [983.0], [1263.0], [1513.0], [1510.0], [1371.0], [1567.0], [1814.0], [1870.0], [2014.0], [1923.0], [1506.0], [1262.0], [1342.0], [1531.0], [1360.0], [1526.0], [1542.0], [1269.0], [1175.0], [1377.0], [1615.0], [1578.0], [1564.0], [1514.0], [1261.0], [1087.0], [1389.0], [1736.0], [1819.0], [1890.0], [1652.0], [1521.0], [1274.0], [1592.0], [1694.0], [1705.0], [1869.0], [1910.0], [1544.0], [1341.0], [1679.0], [1823.0], [1845.0], [1921.0]]


import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ReduceLROnPlateau

# -------------------- Helper Function -----------------------

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
Frame a time series as a supervised learning dataset.
Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
Returns:
		Pandas pd.DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
	# input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out): # forecast sequence (t, t+1, ... t+n)
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan: # drop rows with NaN values
        agg.dropna(inplace=True)
    return agg

# ----------------------------------------------------------------

## load dataset
#dataset = pd.read_pickle("./Poland_transformed_2.pickle")
#values = dataset.values
#
## ensure all data is float
#values = values.astype('float32')
#pd.DataFrame(sth).values == sth!! 

# specify the number of lag hours
look_back = 5
n_features = 1

# frame as supervised learning
reframed = series_to_supervised(sth, look_back, 1)

# split into train and test sets -> coudl use sklearn instead _BUT IT SHUFFLES DATA!
values = reframed.values
split = int(values.shape[0]*0.8)
train = values[:split, :]
test = values[split:, :]

#train_vals = values[:split, :]
#test_vals = values[split:, :]
#
## normalize features after train/test split to avoid information carry through 
#scaler = MinMaxScaler()
#train = scaler.fit_transform(train_vals)
#test = scaler.fit_transform(test_vals)

## split into train and test sets
#vals = reframed.values
#X = vals[:, :-1]
#y = vals[:, -1]
#
## 
#train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0)
#

# split into input and outputs, only predict on Volume 
n_obs = look_back * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features] 
test_X, test_y = test[:, :n_obs], test[:, -n_features]

## reshape input to be 3D [samples, timesteps, features]
#train_X = train_X.reshape((train_X.shape[0], look_back, n_features))
#test_X = test_X.reshape((test_X.shape[0], look_back, n_features))

## design network
#model = Sequential()
#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
#model.add(Dense(1))
#model.compile(loss='mean_squared_error', optimizer='adam')
#
#model.summary() # print model summary
#
## reduce learning rate if loss is constant for some time 
#reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor=0.5,
#                              patience=5, min_lr=0.0001) 
#
## fit network
#history = model.fit(train_X, train_y, epochs=40, batch_size= 10,\
#                    validation_data=(test_X, test_y), verbose=2, \
#                    shuffle = False,callbacks = [reduce_lr])

## plot history
#plt.plot(history.history['loss'], label='train')
#plt.plot(history.history['val_loss'], label='test')
#plt.title('Model Training Losses\n')
#plt.ylabel('Loss')
#plt.xlabel('Epoch')
#plt.legend()
#plt.show()

#Below, we train a regression neural network to classify the data.
from sklearn.neural_network import MLPRegressor
Ann = MLPRegressor(hidden_layer_sizes = [10, 10],alpha=.1,
                   activation='relu', max_iter=5000) #Default is 100 layers, likely "overfitting" although since this problem doesn't have noise, it doesn't really matter
Ann.fit(train_X, train_y)
prediction = Ann.predict(test_X)

print(('{:.1f}\n'*len(prediction)).format(*prediction))

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], look_back*n_features))

# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, -(n_features-1):]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0] # pick the Volume column 

print(('{}'*inv_yhat.shape[0]).format(*inv_yhat))

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, -(n_features-1):]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

# calculate RMSE: mean squared error 
r2 = r2_score(inv_y, inv_yhat)
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test R^2: %.3f\n' % r2)
print('Test sqrt(MSE): %.3f\n' % rmse)


#dataset = pd.DataFrame(sth)
## final plot
#inv_yhat = inv_yhat.reshape(len(inv_yhat),1)
#tmp = np.zeros([dataset.shape[0],1])
#tmp[(dataset.shape[0]- inv_y.shape[0]):] = inv_yhat
#tmp[ tmp==0 ] = np.nan
#Predicted_Vol = tmp
#dataset['Predicted_Series'] = Predicted_Vol # add column to DF
#
#plt.plot(dataset[0], '--',linewidth=0.8, label='Series')
#plt.plot(dataset['Predicted_Series'], 'b', label='Predicted_Series')
#plt.title('Results\n')
#plt.ylabel('Varaible')
#plt.xlabel('Time_steps')
#plt.legend()
#plt.show()


    



