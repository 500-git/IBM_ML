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

df.hist(column = "income", bins = 50)

df.columns

x = df[["region","tenure","age","marital","address","income","ed","employ","retire","gender","reside"]].values
print(x[0:5],"\n")

y = df["custcat"].values
print(y[0:5],"\n")

x = preprocessing.StandardScaler().fit(x).transform(x.astype(float))
print(x[0:5],"\n")

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 4)
print("Train set:",x_train.shape,y_train.shape)
print("Test set:",x_test.shape,y_test.shape)


from sklearn.neighbors import KNeighborsClassifier

print("========= K = 4  =========")
k = 4

neigh = KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)

yhat = neigh.predict(x_test)
print(yhat[0:5],"\n")

from sklearn import metrics
print("Trian set Accuracy: {0}".format(metrics.accuracy_score(y_train,neigh.predict(x_train))))
print("Test set Accuracy: {0}".format(metrics.accuracy_score(y_test,neigh.predict(x_test))))

print("========= K = 6  =========")
k = 6

neigh6 = KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)

yhat6 = neigh6.predict(x_test)
print(yhat6[0:5],"\n")

from sklearn import metrics
print("Trian set Accuracy: {0}".format(metrics.accuracy_score(y_train,neigh6.predict(x_train))))
print("Test set Accuracy: {0}".format(metrics.accuracy_score(y_test,neigh6.predict(x_test))))

Ks = 10

mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks -1))

ConfusionMx = [];

for n in range(1,Ks):
    neigh = KNeighborsClassifier(n_neighbors = n).fit(x_train,y_train)
    yhat = neigh.predict(x_test)
    mean_acc[n - 1] = metrics.accuracy_score(y_test,yhat)

    std_acc[n - 1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

print("mean acc:",mean_acc)

plt.plot(range(1,Ks),mean_acc,"g")
plt.fill_between(range(1,Ks),mean_acc - 1 *std_acc,mean_acc +1 * std_acc,alpha = 0.10)
plt.legend(("Accuracy","+/- 3xstd"))
plt.ylabel("Accuracy")
plt.xlabel("Number of Nabors (K)")
plt.tight_layout()
plt.show()

print("Best accuracy: {0}, best k: {1}".format(mean_acc.max(),mean_acc.argmax()+1))
