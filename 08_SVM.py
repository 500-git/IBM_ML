#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

cell_df = pd.read_csv("cell_samples.csv")
print(cell_df.head(5))

ax = cell_df[cell_df["Class"] == 4][0:50].plot(kind = "scatter",x = "Clump",y = "UnifSize",color = "DarkBlue",label = "malignant");
cell_df[cell_df["Class"] == 2][0:50].plot(kind = "scatter",x = "Clump",y = "UnifSize",color = "Yellow",label = "benign",ax = ax);
plt.show()

print(cell_df.dtypes)

cell_df = cell_df[pd.to_numeric(cell_df["BareNuc"],errors = "coerce").notnull()]
