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
plt.show()

#  plt.scatter(cdf.FUELCONSUMPTION_COMB,cdf.CO2EMISSIONS,color = "blue")
#  plt.xlabel("FUELCONSUMPTION_COMB")
#  plt.ylabel("Emission")
#  plt.show()
#
#  plt.scatter(cdf.FUELCONSUMPTION_COMB,cdf.CO2EMISSIONS,color = "blue")
#  plt.xlabel("Engine Size")
#  plt.ylabel("Emission")
#  plt.show()

plt.scatter(cdf.CYLINDERS,cdf.CO2EMISSIONS,color = "blue")
plt.xlabel("CYLINDERS")
plt.ylabel("Emission")
plt.show()


