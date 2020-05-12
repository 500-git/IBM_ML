#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

#reading data==
df = pd.read_csv("FuelConsumption.csv")
df.head()

cdf = df[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_CITY","FUELCONSUMPTION_HWY","FUELCONSUMPTION_COMB","CO2EMISSIONS"]]
cdf.head(9)

plt.scatter(cdf.ENGINESIZE,cdf.CO2EMISSIONS, color = "blue")
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color = "blue")
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#multiple regression modeling
from sklearn import linear_model
reger = linear_model.LinearRegression()
x = np.asanyarray(train[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB"]])
y = np.asanyarray(train[["CO2EMISSIONS"]])
reger.fit(x,y)
print("Coefficients: {0}".format(reger.coef_))

#prediction
y_hat = reger.predict(test[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB"]])
x = np.asanyarray(test[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB"]])
y = np.asanyarray(test[["CO2EMISSIONS"]])
print("Residual sum of sqrs: {:.2}".format(np.mean(pow(y_hat - y,2))))
print("Variance score: {0}".format(reger.score(x,y)))

#with CITY & HWY
regr2 = linear_model.LinearRegression()
x = np.asanyarray(train[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_CITY","FUELCONSUMPTION_HWY"]])
y = np.asanyarray(train[["CO2EMISSIONS"]])
regr2.fit(x,y)
print("Coefficients: {0}".format(regr2.coef_))

y_hat = regr2.predict(test[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_CITY","FUELCONSUMPTION_HWY"]])
x = np.asanyarray(test[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_CITY","FUELCONSUMPTION_HWY"]])
y = np.asanyarray(test[["CO2EMISSIONS"]])
print("Residual sum of sqrs: {:.2}".format(np.mean(pow(y_hat - y,2))))
print("Variance score: {0}".format(regr2.score(x,y)))

