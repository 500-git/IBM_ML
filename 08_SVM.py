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
#  plt.show()

print(cell_df.dtypes)

cell_df = cell_df[pd.to_numeric(cell_df["BareNuc"],errors = "coerce").notnull()]
cell_df["BareNuc"] = cell_df["BareNuc"].astype("int")
print(cell_df.dtypes)
feature_df = cell_df[["Clump","UnifSize","MargAdh","SingEpiSize","BareNuc","BlandChrom","NormNucl","Mit"]]
x = np.asarray(feature_df)
print(x[0:5])

#  cell_df["Class"] = cell_df["Class"].astype("int")
y = np.asarray(cell_df["Class"])
print(y[0:5])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 4)
print("Train: \n",x_train.shape,"\n",y_train.shape)
print("Test: \n",x_test.shape,"\n",y_test.shape)

from sklearn import svm

clf = svm.SVC(kernel = "rbf")
clf.fit(x_train,y_train)

yhat = clf.predict(x_test)
print(yhat[0:5])

from sklearn.metrics import classification_report,confusion_matrix
import itertools

def plot_confusion_matrix(cm,classes,normalize = False,title = "Confusion Matrix",cmap = plt.cm.Blues):
    if normalize:
        cm = cm.astype("float")/cm.sum(axis = 1)[:,np.newaxis]
        print("Normalized")
    else:
        print("NonNormalized")
    
    print(cm)

    plt.imshow(cm,interpolation = "nearest",cmap = cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation = 45)
    plt.yticks(tick_marks,classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max()/2.
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],fmt),horizontalalignment = "center",color = "white" if cm[i,j] > thresh else "black")

    plt.tight_layout()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

cnf_matrix = confusion_matrix(y_test,yhat,labels = [2,4])
np.set_printoptions(precision = 2)

print(classification_report(y_test,yhat))

plt.figure()
plot_confusion_matrix(cnf_matrix,["Benign(2)","Malignant(4)"])

plt.show()

from sklearn.metrics import f1_score
print("F1_Score: ",f1_score(y_test,yhat,average = "weighted"))

from sklearn.metrics import jaccard_score
print("Jaccard_Score: ",jaccard_score(y_test,yhat,average = "weighted"))

##practice==
#  clf2 = svm.SVC(kernel = "linear")
#  clf2.fit(x_train,y_train)

yhat2 = clf2.predict(x_test)

