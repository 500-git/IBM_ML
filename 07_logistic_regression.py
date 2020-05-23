#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

churn_df = pd.read_csv("ChurnData.csv")
print(churn_df.head(5),"\n")

churn_df = churn_df[["tenure","age","address","income","ed","employ","equip","callcard","wireless","churn"]]
churn_df["churn"] = churn_df["churn"].astype("int")
print(churn_df.head(5),"\n")

print(churn_df.shape)
x = np.asarray(churn_df[["tenure","age","address","income","ed","employ","equip"]])
print(x[0:5],"\n")

y = np.asarray(churn_df[["churn"]])
print(y[0:5],"\n")

x = preprocessing.StandardScaler().fit(x).transform(x)
print(x[0:5],"\n")

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 4)
print("Train set: ",x_train.shape,"\n",y_train.shape,"\n")
print("Test set: ",x_test.shape,"\n",y_test.shape,"\n")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C = 0.01, solver = "liblinear").fit(x_train,y_train.ravel())
print(LR)

yhat = LR.predict(x_test)
print("yhat :",yhat,"\n==================\n")
print("y_test: ",y_test)

yhat_prob = LR.predict_proba(x_test)
print(yhat_prob)

from sklearn.metrics import jaccard_score
print(jaccard_score(y_test,yhat))

from sklearn.metrics import classification_report,confusion_matrix
import itertools

def plot_confusion_matrix(cm,classes,normalize = False,title = "Confusion Matrix",cmap = plt.cm.Blues):
    if normalize:
        cm = cm.astype("float")/cm.sum(axis = 1)[:,np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix,without normalization")

    print(cm,"\n")

    plt.imshow(cm,interpolation = "nearest",cmap = cmap)
    plt.title(title)
    plt.colorbar()
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
print (confusion_matrix(y_test,yhat,labels = [1,0]))

cnf_matrix = confusion_matrix(y_test,yhat,labels = [1,0])
np.set_printoptions(precision = 2)

plt.figure()
plot_confusion_matrix(cnf_matrix,classes = ["churn = 1","churn = 0"],normalize = False,title = "Confusion Matrix")

print(classification_report(y_test,yhat))

from sklearn.metrics import log_loss
print("\n==================\n",log_loss(y_test,yhat_prob))

##practice=====

LR2 = LogisticRegression(C = 0.01,solver = "newton-cg").fit(x_train,y_train.ravel())
yhat = LR2.predict(x_test)
yhat_prob2 = LR2.predict_proba(x_test)

print("\n==================\n",log_loss(y_test,yhat_prob2))

#  print("yhat: \n",yhat,"\n")
#  print("y_test: \n",y_test,"\n")
