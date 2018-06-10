#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import os.path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import datasets
import warnings


# Filter warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

# Generate data
X, y = datasets.make_classification(
    n_samples=10000, n_classes=3, n_clusters_per_class=1)
# Split original dataset to train and test parts.
train_ratio = 0.7
X_train = X[:int(len(X)*train_ratio), :]
y_train = y[:int(len(X)*train_ratio)]
X_test = X[int(len(X)*train_ratio):, :]
y_test = y[int(len(X)*train_ratio):]

log_clf = LogisticRegression(multi_class='multinomial', solver='sag')
rnd_clf = RandomForestClassifier()
svm_clf = SVC(probability=True)

estimators = [('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)]

# Hard voting
print('\n-------Hard Voting-------')
voting_clf = VotingClassifier(estimators=estimators, voting='hard')

# The accuracy of each classifiers
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    #y_pred = clf.predict_proba(X_test)
    #y_pred = np.argmax(y_pred, axis=1)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

print('\n-------Soft Voting-------')
voting_clf.voting = 'soft'
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
