#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

my_data = pd.read_csv("drug200.csv",delimiter = ",")
#  print(my_data[0:5])
#  print(len(my_data),"\n")

x = my_data[["Age","Sex","BP","Cholesterol","Na_to_K"]].values
#  print(x[0:5],"\n")

from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(["F","M"])
x[:,1] = le_sex.transform(x[:,1]) #replace "sex" with int number

le_BP = preprocessing.LabelEncoder()
le_BP.fit(["LOW","NORMAL","HIGH"])
x[:,2] = le_BP.transform(x[:,2]) #replace "BP" with int number

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(["NORMAL","HIGH"])
x[:,3] = le_Chol.transform(x[:,3])
#  print(x[0:5])

y = my_data["Drug"]
#  print(y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 3)
#  print("Train set: ",x_train.shape,"\n",y_train.shape,"\n")
#  print("Test set: ",x_test,y_test,"\n")
#  print(y_test.values)

drugTree = DecisionTreeClassifier(criterion = "entropy",max_depth = 4)
#  print(drugTree)

drugTree.fit(x_train,y_train)

predTree = drugTree.predict(x_test)

#  print(predTree[0:10])
#  print(y_test[0:10])

from sklearn import metrics

import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ",metrics.accuracy_score(y_test,predTree))

correct = 0
for x in range(0,len(predTree)):
    if predTree[x] == y_test.values[x]:
        correct += 1

print("manual acc_score = ",correct/len(predTree))

from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out = tree.export_graphviz(drugTree,feature_names = featureNames,out_file = dot_data,class_names = np.unique(y_train),filled = True,special_characters = True,rotate = False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize = (100,200))
plt.imshow(img,interpolation = "nearest")
