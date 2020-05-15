#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import numpy as np
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
from matplotlib.ticker import NullFormatter
from sklearn import preprocessing

df = pd.read_csv("teleCust1000t.csv")
df.head()

print(df["custcat"].value_counts())

df.hist(colum = "income", bins = 50)

