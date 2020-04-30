#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use("TKAgg")
import pandas as pd
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
#  %matplotlib inline

df = pd.read_csv("FuelConsumption.csv")
df.head()

df.describe()

cdf = df[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB","CO2EMISSIONS"]]
cdf.head(9)

viz =  cdf[["CYLINDERS","ENGINESIZE","CO2EMISSIONS","FUELCONSUMPTION_COMB"]]
viz.hist()
#  plt.show()

plt.scatter(cdf.FUELCONSUMPTION_COMB,cdf.CO2EMISSIONS,color = "blue")
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.FUELCONSUMPTION_COMB,cdf.CO2EMISSIONS,color = "blue")
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()
#
plt.scatter(cdf.CYLINDERS,cdf.CO2EMISSIONS,color = "blue")
plt.xlabel("CYLINDERS")
plt.ylabel("Emission")
plt.show()

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color = "blue")
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()

from sklearn import linear_model

reger = linear_model.LinearRegression()
train_x = np.asanyarray(train[["ENGINESIZE"]])
train_y = np.asanyarray(train[["CO2EMISSIONS"]])
reger.fit(train_x,train_y)

print("Coefficients: ",reger.coef_)
print("Intercept: ",reger.intercept_)

plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color = "blue")
plt.plot(train_x,reger.coef_[0][0]*train_x + reger.intercept_[0],"-r")
plt.xlabel("Engine Size")
plt.ylabel("Emission")

from sklearn.metrics import r2_score

test_x = np.asanyarray(test[["ENGINESIZE"]])
test_y = np.asanyarray(test[["CO2EMISSIONS"]])
test_y_hat = reger.predict(test_x)

print("Mean absolute error: {0}".format(np.mean(np.absolute(test_y_hat - test_y))))
print("Residual sum of squares (MES): {0}".format(np.mean(test_y_hat - test_y) ** 2))
print("R2-score: {0}".format(r2_score(test_y_hat,test_y)))
