import pandas

# data = pandas.read_csv('titanic.csv', index_col='PassengerId')
# print(data['Sex'].value_counts()) #female male
# print('Survived ', (data.size / data['Survived'].value_counts()[1]).round(2)) #survived pasengers
# print('Survived ', ((data['Survived'].value_counts()[1] / data['Survived'].size)*100).round(2)) #survived pasengers
# print(((data['Pclass'].value_counts()[1] / data.size)*100).round(2))  #first class from all rounded
# print(((data['Pclass'].value_counts()[1] / data['Pclass'].size)*100).round(2))  #first class from all rounded
# print((data['Age'].median())) #median age
# print((data['Age'].mean()).round(2)) #averange age
# print((data['SibSp'].corr(data['Parch'])).round(2)) #pearce corelation
# print(data[data['Sex'] == 'female']['Name']) #Selected all female names
# print((data[data['Sex'] == 'female']['Name']).str.split(',').str[1].value_counts())

# import sklearn.neighbors.KNeighborsClassifier as neibourClass
#####################################################################
# import pandas as pd
# import numpy as np
# from sklearn import metrics
# from sklearn.neighbors import KNeighborsClassifier
#
# data = np.loadtxt(r'D:\\MyPython\\wine.data', delimiter=",")
# X = data[:, 1:14]
# y = data[:, 0]
#
# kf = KFold(len(data), n_folds=5, shuffle=True, random_state=42)
# knn = neighbors.KNeighborsClassifier()
#
# knn.n_neighbors = icvs = cross_val_score(estimator=knn, X=X, y=y, cv=kf, scoring='accuracy')
# mean = cvs.mean()
#
# data_scale = preprocessing.scale(data)
# X = data_scale[:, 1:14]
# print(X)
######################################################################
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import sklearn

data = pandas.read_csv('wine.data', sep=',', header=None)  # Load data with separators

y = data[0]  # separate data and answers
x = data.drop(0, axis=1)  # drop answers

kf = KFold(n_splits=5, shuffle=True, random_state=42)  # k-fold cross validation  separates into n(5)_splits, algorithm is trained on n-1(4) splits and tested on n-th(5th) split, this is done 5 times

maxNeib = 0  # temp for max neibours
maxMean = 0  # temp for max accuracy

x = sklearn.preprocessing.scale(x)  # normalization normirovanie

for c in range(1, 51):  # cycle for finding better value
    nbrs = KNeighborsClassifier(n_neighbors=c, algorithm='auto')  # defining machine learning algorithm
    cvs = cross_val_score(estimator=nbrs, cv=kf, X=x, y=y)  # counting cross validation score
    result = cvs.mean().round(2)  # rounding
    print('With neighbours = ', c, 'mean =', result, )  # printing

    if result > maxMean:  # finding best result with max accuracy
        maxNeib = c
        maxMean = result

print('Max mean is', maxMean, 'with neibours', maxNeib)  # printing best result
