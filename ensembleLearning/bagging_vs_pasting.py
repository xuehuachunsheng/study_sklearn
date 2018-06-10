#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score

X, y = datasets.make_classification(n_samples=5000, n_classes=5, n_clusters_per_class=1, n_informative=3)

splitRatio = 0.7
X_train = X[:int(splitRatio*len(X)), :]
y_train = y[:int(splitRatio*len(X))]
X_test = X[int(splitRatio*len(X)):, :]
y_test = y[int(splitRatio*len(X)):]

# bootstrap=True represents Bagging and bootstrap=False represents Pasting
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=50, max_samples=1., bootstrap=True, n_jobs=-1, oob_score=True)
past_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=50, max_samples=1., bootstrap=False, n_jobs=-1)

# Bagging prediction
print('Bagging classifier')
bag_clf.fit(X_train, y_train)
y_pred_bag = bag_clf.predict(X_test)
print(bag_clf.__class__.__name__, ', ACC: ', accuracy_score(y_pred_bag, y_test))

# Pasting prediction
print('Pasting classifier')
past_clf.fit(X_train, y_train)
y_pred_past = past_clf.predict(X_test)
print(past_clf.__class__.__name__, ', ACC: ', accuracy_score(y_pred_past, y_test))


