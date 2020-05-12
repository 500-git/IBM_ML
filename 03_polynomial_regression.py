#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pylab as pl

df = pd.read_csv("FuelConsumption.csv")
df.head()

cdf = df[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB","CO2EMISSIONS"]]
cdf.head(9)

plt.scatter(cdf.ENGINESIZE,cdf.CO2EMISSIONS, color = "blue")
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
train_x = np.asanyarray(train[["ENGINESIZE"]])
train_y = np.asanyarray(train[["CO2EMISSIONS"]])

test_x = np.asanyarray(test[["ENGINESIZE"]])
test_y = np.asanyarray(test[["CO2EMISSIONS"]])

poly = PolynomialFeatures(degree = 2)
train_x_poly = poly.fit_transform(train_x) # matrix x_n^0 x_n^1 x_n^2
train_x_poly

clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly,train_y)
print("Coefficients: ",clf.coef_)
print("Intercept: ",clf.intercept_)

plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS, color = "blue")
xx = np.arange(0.0,10.0,0.1)
yy = clf.intercept_[0] + xx * clf.coef_[0][1] + np.power(xx,2) * clf.coef_[0][2]
plt.plot(xx,yy, "-r")
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

from sklearn.metrics import r2_score

test_x_poly = poly.fit_transform(test_x)
test_y_ = clf.predict(test_x_poly)

print("Mean absolute error: ",np.mean(np.absolute(test_y_ - test_y)))
print("Mean sum of squares: ",np.mean((test_y_ - test_y) ** 2))
print("R2_score: ",r2_score(test_y_,test_y)) 

#cubic
poly3 = PolynomialFeatures(degree = 3)
train_x_poly3 = poly3.fit_transform(train_x)
train_x_poly3

clf3 = linear_model.LinearRegression()
train_y3_ = clf3.fit(train_x_poly3,train_y)
print("Cubic Eoefficients: ",clf3.coef_)
print("Cubic Intercept: ",clf3.intercept_)

plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS, color = "blue")
xx3 = np.arange(0.0,10.0,0.1)
yy3 = clf3.intercept_[0] + xx3 * clf3.coef_[0][1] + np.power(xx3,2) * clf3.coef_[0][2] + np.power(xx3,3) * clf3.coef_[0][3]
plt.plot(xx3,yy3,"-r")
plt.xlabel("Engine size_3")
plt.ylabel("Emission_3")
plt.show()

test_x_poly3 = poly3.fit_transform(test_x)
test_y3_ = clf3.predict(test_x_poly3)


print("Cubic Mean absolute error: ",np.mean(np.absolute(test_y3_ - test_y)))
print("Cubic Mean sum of squares: ",np.mean((test_y3_ - test_y) ** 2))
print("Cubic R2_score: ",r2_score(test_y3_,test_y)) 
