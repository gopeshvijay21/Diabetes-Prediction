# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 19:12:53 2018

@author: gopesh
"""
# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

# importing the dataset
dataset = pd.read_csv('diabetes.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# spliting the dataset into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# fitting the Logistic Regression classifier to the training set
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(random_state = 0)
classifier_lr.fit(X_train, y_train)

# predicting the test set results from Logistic Regression Classifier
y_pred_lr = classifier_lr.predict(X_test)

# making the confusion matrix from Logistic Regression classifier
cm_lr = confusion_matrix(y_test, y_pred_lr)

# Fitting the k-nn classifier to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors = 5,metric = 'minkowski', p = 2)
classifier_knn.fit(X_train, y_train)

# predicting the test set results from k-nn classifier
y_pred_knn = classifier_knn.predict(X_test)

# making the confusion matrix from knn classifier
cm_knn = confusion_matrix(y_test, y_pred_knn)

# Fitting the linear SVM classifier to the training set
classifier_linear_svm = SVC(kernel = 'linear', random_state = 0)
classifier_linear_svm.fit(X_train, y_train)

# predicting the test set results from linear SVM classifier
y_pred_linear_svm = classifier_linear_svm.predict(X_test)

# making the confusion matrix from linear SVM classifier
cm_linear_svm = confusion_matrix(y_test, y_pred_linear_svm)

# Fitting the polynomial SVM classifier to the training set
classifier_poly_svm = SVC(kernel = 'poly', random_state = 0)
classifier_poly_svm.fit(X_train, y_train)

# predicting the test set results from polynomial SVM classifier
y_pred_poly_svm = classifier_poly_svm.predict(X_test)

# making the confusion matrix from polynomial SVM classifier
cm_poly_svm = confusion_matrix(y_test, y_pred_poly_svm)

# Fitting the gaussian rbf SVM classifier to the training set
classifier_rbf_svm = SVC(kernel = 'rbf', random_state = 0)
classifier_rbf_svm.fit(X_train, y_train)

# predicting the test set results from gaussian rbf SVM classifier
y_pred_rbf_svm = classifier_rbf_svm.predict(X_test)

# making the confusion matrix from gaussian rbf SVM classifier
cm_rbf_svm = confusion_matrix(y_test, y_pred_rbf_svm)

# Fitting the sigmoid SVM classifier to the training set
classifier_sigmoid_svm = SVC(kernel = 'sigmoid', random_state = 0)
classifier_sigmoid_svm.fit(X_train, y_train)

# predicting the test set results from sigmoid SVM classifier
y_pred_sigmoid_svm = classifier_sigmoid_svm.predict(X_test)

# making the confusion matrix from sigmoid SVM classifier
cm_sigmoid_svm = confusion_matrix(y_test, y_pred_rbf_svm)

# Fitting the Naive Bayes classifier to the training set
from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)

# predicting the test set results from Naive Bayes classifier
y_pred_nb = classifier_nb.predict(X_test)

# making the confusion matrix from Naive Bayes classifier
cm_nb = confusion_matrix(y_test, y_pred_nb)

# Fitting the Decision Tree classifier to the training set
from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_dt.fit(X_train, y_train)

# predicting the test set results from Decision Tree classifier
y_pred_dt = classifier_dt.predict(X_test)

# making the confusion matrix from Decision Tree classifier
cm_dt = confusion_matrix(y_test, y_pred_dt)

# Fitting the Random Forest classifier to the training set
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_rf.fit(X_train, y_train)

# predicting the test set results from Random Forest classifier
y_pred_rf = classifier_rf.predict(X_test)

# making the confusion matrix from Random Forest classifier
cm_rf = confusion_matrix(y_test, y_pred_rf)