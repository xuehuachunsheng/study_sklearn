#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score

X, y = datasets.make_classification(n_samples=5000, n_classes=5, n_clusters_per_class=1, n_informative=3)

splitRatio = 0.7
X_train = X[:int(splitRatio*len(X)), :]
y_train = y[:int(splitRatio*len(X))]
X_test = X[int(splitRatio*len(X)):, :]
y_test = y[int(splitRatio*len(X)):]

rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(X_train, y_train)
y_pred = rnd_clf.predict(X_test)

print(rnd_clf.__class__.__name__, accuracy_score(y_pred, y_test))