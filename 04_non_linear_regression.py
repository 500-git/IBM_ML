#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

df = pd.read_csv("china_gdp.csv")
print(df.head(10))


plt.figure(figsize=(8,5))
x_data,y_data = (df["Year"].values,df["Value"].values)
plt.plot(x_data,y_data,"ro")
plt.xlabel("Year")
plt.ylabel("GDP")
plt.show()

#  X = np.arange(-5.0,5.0,0.1)
#  Y = 1./(1. + np.exp(-X))
#
#  plt.plot(X,Y)
#  plt.xlabel("Dep. Var.")
#  plt.ylabel("Indep. Var.")
#  plt.show()

def sigmoid(x,beta1,beta2):
    y = 1/(1 + np.exp(-(beta1) * (x - beta2)))
    return y

beta_1 = 0.1
beta_2 = 1990.

Y_pred = sigmoid(x_data,beta_1,beta_2)

plt.plot(x_data,Y_pred * 15000000000000.)
plt.plot(x_data,y_data,"ro")

xdata =  x_data/max(x_data)
ydata = y_data/max(y_data)

from scipy.optimize import curve_fit
p_opt,p_cov = curve_fit(sigmoid,xdata,ydata)
print("beta1 = {0}   beta2 = {1}".format(p_opt[0],p_opt[1]))

x = np.linspace(1960,2005,55)
x = x/max(x)
plt.figure(figsize = (8,5))
y = sigmoid(x,*p_opt)
plt.plot(xdata,ydata,"ro",label = "data")
plt.plot(x,y,linewidth = 3.0,label = "fit")
plt.legend(loc = "best")
plt.xlabel("Year")
plt.ylabel("GDP")
plt.show()
