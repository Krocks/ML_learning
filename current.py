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
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.neighbors import KNeighborsClassifier
# import sklearn
#
# data = pandas.read_csv('wine.data', sep=',', header=None)  # Load data with separators
#
# y = data[0]  # separate data and answers
# x = data.drop(0, axis=1)  # drop answers
#
# kf = KFold(n_splits=5, shuffle=True, random_state=42)  # k-fold cross validation  separates into n(5)_splits, algorithm is trained on n-1(4) splits and tested on n-th(5th) split, this is done 5 times
#
# maxNeib = 0  # temp for max neibours
# maxMean = 0  # temp for max accuracy
#
# x = sklearn.preprocessing.scale(x)  # normalization normirovanie
#
# for c in range(1, 51):  # cycle for finding better value
#     nbrs = KNeighborsClassifier(n_neighbors=c, algorithm='auto')  # defining machine learning algorithm
#     cvs = cross_val_score(estimator=nbrs, cv=kf, X=x, y=y)  # counting cross validation score
#     result = cvs.mean().round(2)  # rounding
#     print('With neighbours = ', c, 'mean =', result, )  # printing
#
#     if result > maxMean:  # finding best result with max accuracy
#         maxNeib = c
#         maxMean = result
#
# print('Max mean is', maxMean, 'with neibours', maxNeib)  # printing best result
##############################################################################
# from sklearn.datasets import load_boston
# from sklearn.preprocessing import scale
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# import numpy
#
# boston = load_boston()
# data = boston['data']
# target = boston['target']
#
# data = scale(data)
#
# steps = numpy.linspace(1.0, 10.0, num=200)
#
# minError = -100
# iError = 0
# for i in steps:
#     nbrs = KNeighborsRegressor(n_neighbors=5, weights='distance', p=i)
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     cvs = cross_val_score(estimator=nbrs, cv=kf, X=data, y=target, scoring='neg_mean_squared_error')
#     result = cvs.mean().round(2)
#     print(result)
#     if result > minError:
#         minError = result
#         iError = i
# print('MinError is ', minError, 'and i is', iError)

# import numpy as np
# import pandas
# from sklearn.linear_model import Perceptron
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler
#
# train = pandas.read_csv('perceptron-train.csv', header=None)
# test = pandas.read_csv('perceptron-test.csv', header=None)
#
#
# X_train = train[train.columns[1:4]]
# y_train = train[train.columns[0]]
#
# X_test = test[test.columns[1:4]]
# y_test = test[test.columns[0]]
#
# clf = Perceptron(random_state=241)
# clf.fit(X_train, y_train)
#
# predictions = clf.predict(X_test)
#
# scaler = StandardScaler()
#
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# clf2 = Perceptron(random_state=241)
# clf2.fit(X_train_scaled, y_train)
#
# predictions_scaled = clf2.predict(X_test_scaled)
#
#
# result_not_normalized = accuracy_score(y_test, predictions)
# result_normalized = accuracy_score(y_test, predictions_scaled)
#
# # print(predictions)
# print('Not normalized result of accuracy on test dataset is ', result_not_normalized)
# print('Normalized result of accuracy on test dataset is ', result_normalized)
# print('Result is ', (result_normalized - result_not_normalized).round(3))
###################################################################
# import pandas
# from sklearn.svm import SVC
#
# data = pandas.read_csv('svm-data.csv', header=None)
# y = data[0]
# X = data[[1, 2]]
# clr = SVC(random_state=241, C=100000, kernel='linear')
# clr.fit(X, y)
#
# print(clr.support_)
##################################################################
# from sklearn import datasets
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC
# import numpy as np
# newsgroups = datasets.fetch_20newsgroups(
#     subset='all',
#     categories=['alt.atheism', 'sci.space']
# )  # data - text , target class of text ?
#
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(newsgroups.data)  # with target?
# y = newsgroups.target
# # X = vectorizer.fit_transform(newsgroups.data, y=newsgroups.target)
# # idf = vectorizer.idf_
# # feature_mapping = vectorizer.get_feature_names()
#
# # grid = {'C': np.power(10.0, np.arange(-5, 6))}   # grid given as example
# # cv = KFold(n_splits=5, shuffle=True, random_state=241)
# # clf = SVC(kernel='linear', random_state=241)
# # gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
# # gs.fit(X, y)
#
# # diff_C = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0, 10**-1, 10**-2, 10**-3, 10**4, 10**5] # Looking for max C
# # for C in diff_C:
# #     clr = SVC(random_state=241, C=C, kernel='linear')
# #     # clr.fit(X, y)
# #     cv = KFold(n_splits=5, shuffle=True, random_state=241)
# #     cvs = cross_val_score(estimator=clr, cv=cv, X=X, y=y, scoring='accuracy')
# #     print('C is ', C, 'and cvs is ', cvs.mean())
#
# clr = SVC(random_state=241, C=1, kernel='linear')
# clr.fit(X, y)
#
# ind = np.argsort(np.absolute(np.asarray(clr.coef_.todense())).reshape(-1))[-10:]
# print(ind)
#
# words = [vectorizer.get_feature_names()[i] for i in ind]
# print(sorted(words))
# # with open("q1.txt", "w") as output:
# #     output.write('%s' % (" ".join(sorted(words))))
# ######################################################################################
# import pandas
# import numpy as np
# from sklearn.metrics import roc_auc_score
# from scipy.spatial import distance
#
# def split(data):
#     X = data.iloc[:,1:].values
#     y = data.iloc[:,0].values
#     return X, y
#
# def sigmoid(X, w): # for roc_auc ~ qality score
#     return 1 / (1 + np.exp(-np.dot(X, w)))
#
# def cost(X, y, w, C):
#     sum = 0
#     n = X.shape[0]
#     m = X.shape[1]
#     for i in range(n):
#         sum += np.log(1 + np.exp(-y[i] * np.dot(X[i], w)))
#     reg = C * (w ** 2).sum() / m
#     cost = sum / np.double(n) + reg
#     return cost
#
# def train(X, y, k, C):
#     n = X.shape[0]
#     m = X.shape[1]
#     w = np.zeros(m)
#     c = cost(X, y, w, C)
#     threshold = 1e-5
#     for iteration in range(10000):
#         new_w = np.zeros(m)
#         for j in range(m):
#             sum = 0
#             for i in range(n):
#                 sum += y[i] * X[i, j] * (1 - 1 / (1 + np.exp(-y[i] * np.dot(X[i], w))))
#             new_w[j] = w[j] + k * sum / np.double(n) - k * C * w[j]
#         new_cost = cost(X, y, new_w, C)
#         if distance.euclidean(w, new_w) <= threshold:
#             return new_w
#         c = new_cost
#         w = new_w
#     return w
#
# data = pandas.read_csv('data-logistic.csv', header=None)
# X, y = split(data)
# k = 0.1
# score = roc_auc_score(y, sigmoid(X, train(X, y, k, C = 0)))
# score_reg = roc_auc_score(y, sigmoid(X, train(X, y, k, C = 10)))  # C - коефициент регуляризации
#
# print("SCORE", score.round(3))
# print("SCORE REG", score_reg.round(3))
# with open("q1.txt", "w") as output:
#     output.write('%.3f %.3f' % (score, score_reg))

import pandas
import sklearn.metrics

# data = pandas.read_csv('classification.csv')
#
#
# def sort():
#
#     t = data[data['true'] == data['pred']]['pred']
#     tp = t.sum()
#     tn = t.size - tp
#
#     f = data[data['true'] != data['pred']]['pred']
#     fp = f.sum()
#     fn = f.size - fp
#
#     print('%d %d %d %d' % (tp, fp, fn, tn))
#
# #
# # def count():
# #     print('Accuracy is', (sklearn.metrics.accuracy_score(y_true=data['true'], y_pred=data['pred'])).round(2))
# #     print('Precious is', (sklearn.metrics.precision_score(y_true=data['true'], y_pred=data['pred'])).round(2))
# #     print('Recall is', (sklearn.metrics.recall_score(y_true=data['true'], y_pred=data['pred'])).round(2))
# #     print('F-Value is', (sklearn.metrics.f1_score(y_true=data['true'], y_pred=data['pred'])).round(2))
#
#
# sort()
# count()
# print(data)

# fit_data = pandas.read_csv('scores.csv')
#
# def count_score_logreg():
#     precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true=fit_data['true'], probas_pred=fit_data['score_logreg'])
#     max_precious = 0
#     for i in range (0, thresholds.size):
#         if recall[i] > 0.7:
#             if precision[i] > max_precious:
#                 max_precious = precision[i]
#     print('score_logreg', max_precious)
#
# def count_score_svm():
#     precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true=fit_data['true'], probas_pred=fit_data['score_svm'])
#     max_precious = 0
#     for i in range (0, thresholds.size):
#         if recall[i] > 0.7:
#             if precision[i] > max_precious:
#                 max_precious = precision[i]
#     print('score_svm', max_precious)
#
# def count_score_knn():
#     precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true=fit_data['true'], probas_pred=fit_data['score_knn'])
#     max_precious = 0
#     for i in range (0, thresholds.size):
#         if recall[i] > 0.7:
#             if precision[i] > max_precious:
#                 max_precious = precision[i]
#     print('score_knn', max_precious)
#
# def count_score_tree():
#     precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true=fit_data['true'], probas_pred=fit_data['score_tree'])
#     max_precious = 0
#     for i in range (0, thresholds.size):
#         if recall[i] > 0.7:
#             if precision[i] > max_precious:
#                 max_precious = precision[i]
#     print('score_tree', max_precious)
#
#
# def count_scores():
#     count_score_logreg()
#     count_score_knn()
#     count_score_svm()
#     count_score_tree()
#
# # count_scores()
# sort()
#
#
# for i in range(1, 5):
#     print(fit_data.axes[1][i])
#     print((sklearn.metrics.roc_auc_score(y_true=fit_data['true'], y_score=fit_data[fit_data.axes[1][i]])).round(2))
#
# print('score_logreg is', (sklearn.metrics.roc_auc_score(y_true=fit_data['true'], y_score=fit_data['score_logreg'])).round(2))
# print('score_svm is ', (sklearn.metrics.roc_auc_score(y_true=fit_data['true'], y_score=fit_data['score_svm'])).round(2))
# print('score_knn is ', (sklearn.metrics.roc_auc_score(y_true=fit_data['true'], y_score=fit_data['score_knn'])).round(2))
# print('score_tree is ', (sklearn.metrics.roc_auc_score(y_true=fit_data['true'], y_score=fit_data['score_tree'])).round(2))
# print(fit_data)
# print(data[0:1])
# print(data.shape[1])
import pandas

# import pandas
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction import DictVectorizer
# from scipy.sparse import hstack
# from sklearn.linear_model import Ridge
#
# data_train = pandas.read_csv('salary-train.csv')
# data_test = pandas.read_csv('salary-test-mini.csv')
# y = data_train['SalaryNormalized']
#
# data_train['FullDescription'] = data_train['FullDescription'].str.lower()
# data_train['LocationNormalized'] = data_train['LocationNormalized'].str.lower()
#
# data_train['FullDescription'] = data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
# data_train['LocationNormalized'].fillna('nan', inplace=True)
# data_train['ContractTime'].fillna('nan', inplace=True)
#
# vectorizer = TfidfVectorizer(min_df=5)
# X_tfid_train = vectorizer.fit_transform(data_train['FullDescription'])
# X_tfid_test = vectorizer.transform(data_test['FullDescription'])
#
# enc = DictVectorizer()
# X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
# X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))
#
# contcatenate_train_data = hstack((X_tfid_train, X_train_categ))
# contcatenate_test_data = hstack((X_tfid_test, X_test_categ))
#
# ridge = Ridge(alpha=1, random_state=241)
# ridge.fit(X=contcatenate_train_data, y=y)
#
# print((ridge.predict(contcatenate_test_data)).round(2))

# import pandas
# from numpy import corrcoef, argmax
# from sklearn.decomposition import PCA
#
# dou_jones = pandas.read_csv('djia_index.csv')['^DJI']
# close_prices = pandas.read_csv('close_prices.csv').iloc[:, 1:]
#
# pca = PCA(n_components=10)
# pca.fit(close_prices)
# pc = pca.transform(close_prices)[:, 0]
#
#
# dispersion = 0
# components = 0
# for variable in pca.explained_variance_ratio_:
#     components += 1
#     dispersion += variable
#     print('components ', components, 'dispersion', dispersion, 'variable number', variable)
#     if dispersion >= 0.9:
#         break
# print(1, components)
#
# correlation = corrcoef(pc, dou_jones)[0, 1]
# print(correlation.round(2))


# import pandas
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import KFold, cross_val_score
# import numpy as np
# from sklearn.metrics import r2_score
#
# data = pandas.read_csv('abalone.csv')
# data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
# y = data['Rings']
# X = data.iloc[:, 0:8]
# # X = data.iloc[:, :-1]
#
# kf = KFold(n_splits=5, shuffle=True, random_state=1)
# for i in range(1, 50):
#     rfr = RandomForestRegressor(random_state=1, n_estimators=i)
#     rfr.fit(X=X, y=y)
#     cvs = cross_val_score(estimator=rfr, cv=kf, X=X, y=y, scoring='r2')
#     print('Trees = ', i, 'Score = ', cvs.mean())
######################################################################################################
# https://www.coursera.org/learn/vvedenie-mashinnoe-obuchenie/programming/uDU31/gradiientnyi-bustingh-nad-rieshaiushchimi-dieriev-iami
import pandas
import numpy as np, os
import math
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
if not os.path.exists('plots'):
    os.makedirs('plots')


def plot(train_loss, test_loss, fname):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # %matplotlib inline
    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.savefig(fname)


data = pandas.read_csv('gbm-data.csv')
y = data.iloc[:, 0]
X = data.iloc[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)
learning_rate = [1, 0.5, 0.3, 0.2, 0.1]
for i in range(0, len(learning_rate)):
    print('Learning rate = ', learning_rate[i])
    clf = GradientBoostingClassifier(n_estimators=250, verbose=2, random_state=241, learning_rate=learning_rate[i])
    clf.fit(X_train, y_train)
    error_train = clf.staged_decision_function(X_train)
    error_test = clf.staged_decision_function(X_test)
    print(error_train)
    # y_pred_test = clf.predict(X_test)
    # y_pred_train = clf.predict(X_train)
    # y_pred_test = 1 / (1 + np.exp(-error_test))
    # y_pred_train = 1 / (1 + np.exp(-error_train))
    # stg_loss_test = log_loss(y_pred=y_pred_test, y_true=y_test)
    # stg_loss_train = log_loss(y_pred=y_pred_train, y_true=y_train)
    # print(stg_loss_test)

#     # plot(stg_loss_train, stg_loss_test, 'plots/%.1f.png' % (learning_rate[i]))
# from pandas import read_csv
# data = read_csv('gbm-data.csv')
#
# X = data.iloc[:,1:]
# y = data.iloc[:,0]
#
# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)
#
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import log_loss
# import numpy as np, os
# if not os.path.exists('plots'):
#     os.makedirs('plots')
#
# def plot(train_loss, test_loss, fname):
#     import matplotlib
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt
#     # %matplotlib inline
#     plt.figure()
#     plt.plot(test_loss, 'r', linewidth=2)
#     plt.plot(train_loss, 'g', linewidth=2)
#     plt.legend(['test', 'train'])
#     plt.savefig(fname)
#
# min_losses = {}
# for index, learning_rate in enumerate([1, 0.5, 0.3, 0.2, 0.1], start=1):
#     clf = GradientBoostingClassifier(n_estimators=250, learning_rate=learning_rate, verbose=True, random_state=241)
#     clf.fit(X_train, y_train)
#     train_pred_iters = clf.staged_predict_proba(X_train)
#     test_pred_iters = clf.staged_predict_proba(X_test)
#     train_loss = [ log_loss(y_train, pred) for pred in train_pred_iters]
#     test_loss = [ log_loss(y_test, pred) for pred in test_pred_iters]
#     best_iter = np.argmin(test_loss)
#     min_losses[learning_rate] = (test_loss[best_iter], best_iter)
#     plot(train_loss, test_loss, 'plots/%d_%.1f.png' % (index, learning_rate))
#
# print('Min losses ',min_losses[0.2])
#
# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier(n_estimators=min_losses[0.2][1], random_state=241)
# rf.fit(X_train, y_train)
# rf_pred = rf.predict_proba(X_test)[:, 1]
# rf_score = log_loss(y_test, rf_pred)
# print('Question 3', rf_score)