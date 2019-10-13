#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 16:27:00 2018

Multi-lable Classification with sklearn

https://www.analyticsvidhya.com/blog/2017/08/introduction-to-multi-label-classification/

4. Techniques for Solving a Multi-Label classification problem
Basically, there are three methods to solve a multi-label classification problem, namely:

Problem Transformation:
    - Binary Relevance; predict eatch Y label separately, so if say have 3 poss Y labels tarin 3 models 
    
    - Classifier Chains; predict Y_1 on X data, predict Y-2 on x data and Y_1 column ,pred Y-3 on X data 
                        and all prev Y columns etc.. This is quite similar to binary relevance, the only 
                        difference being it forms chains in order to preserve label correlation. 
                        
    - Label Powerset: transform the problem into a multi-class problem with one multi-class classifier is 
                        trained on all unique label combinations found in the training data. So say out of Y1-3
                        say we have 7 unique combinations of lables appear in the data. so we make one y target with 
                        lablencoded target values form 0-7
                        
Adapted Algorithm: Adapted algorithm, as the name suggests, adapting the algorithm to directly perform multi-label 
                    classification, rather than transforming the problem into different subsets of problems.
                    
Ensemble approaches:

"""
############################# general ####################################
# gerate data 
from sklearn.datasets import make_multilabel_classification

# this will generate a random multi-label dataset
#X, y = make_multilabel_classification(sparse = True, n_labels = 20,
#return_indicator = 'sparse', allow_unlabeled = False)

X, y = make_multilabel_classification(n_samples=100, n_features=20, n_classes=4,
                                      n_labels=2, length=50, allow_unlabeled = False,
                                      sparse=False, return_indicator='dense',
                                      return_distributions=False, random_state=0)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

########################## binary relevance #########################################
# using binary relevance
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB

# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
classifier = BinaryRelevance(GaussianNB())

# train
classifier.fit(X_train, y_train)

# predict
predictions = classifier.predict(X_test)

from sklearn.metrics import accuracy_score , classification_report
#from sklearn.metrics import confusion_matrix

accuracy_score(y_test, predictions)
#confusion_matrix(y_test, predictions)
print(classification_report(y_test, predictions))

########################## classifier chains #######################################
# using classifier chains
from skmultilearn.problem_transform import ClassifierChain
from sklearn.naive_bayes import GaussianNB

# initialize classifier chains multi-label classifier
# with a gaussian naive bayes base classifier
classifier = ClassifierChain(GaussianNB())

# train
classifier.fit(X_train, y_train)

# predict
predictions = classifier.predict(X_test)

from sklearn.metrics import accuracy_score , classification_report

accuracy_score(y_test, predictions)
print(classification_report(y_test, predictions))
'''
We can see that using this we obtained an accuracy of about 21%, which is very
 less than binary relevance. This is maybe due to the absence of label correlation 
 since we have randomly generated the data.
 '''
########################## Label Powerset #######################################
 # using Label Powerset
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB

# initialize Label Powerset multi-label classifier
# with a gaussian naive bayes base classifier
classifier = LabelPowerset(GaussianNB())

# train
classifier.fit(X_train, y_train)

# predict
predictions = classifier.predict(X_test)

from sklearn.metrics import accuracy_score , classification_report
accuracy_score(y_test,predictions)
print(classification_report(y_test, predictions))
 
 # BEST SCORE! so far 
 
 #####################################################################################
#################### Adapted Algorithms ##################################
 ######################################################################################################
"""
 multi-label version of kNN is represented by MLkNN. So, let us quickly 
 mplement this on our randomly generated data set.
 
"""
from skmultilearn.adapt import MLkNN

classifier = MLkNN(k=5)

# train
classifier.fit(X_train, y_train)

# predict
predictions = classifier.predict(X_test)

from sklearn.metrics import accuracy_score , classification_report
print('accuracy:',  accuracy_score(y_test,predictions))
print(classification_report(y_test, predictions))



################################################################################
#################### MultiOutputClassifier/ Regressor ##################################
################################################################################

'''
Multioutput classification/ regression support can be added to any classifier with 
MultiOutputClassifier/ MultiOutputRegressor. This strategy consists of fitting one 
classifier per target. This allows multiple target variable classifications.
 The purpose of this class is to extend estimators to be able to estimate a series of 
 target functions (f1,f2,f3…,fn) that are trained on a single X predictor matrix to
 predict a series of responses (y1,y2,y3…,yn).
'''
######################## MultiOutputRegressor ###############################
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

X, y = make_regression(n_samples=100, n_targets=3, random_state=1)

#gbr = GradientBoostingRegressor(random_state=0)
#
#MultiOutputRegressor(gbr).fit(X, y).predict(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

gbr = GradientBoostingRegressor(random_state=0)

gbr_fitted = MultiOutputRegressor(gbr).fit(X_train, y_train)

predictions = gbr_fitted.predict(X_test)

from sklearn.metrics import r2_score , mean_squared_error

print('mse:',mean_squared_error(y_test, predictions))
print('r2:',r2_score(y_test, predictions))

######################## MultiOutputClassifier ###############################

from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import numpy as np
# WITH DATA MADE FROM SCRATCH 
#X, y1 = make_classification(n_samples=10, n_features=100, n_informative=30, n_classes=3, random_state=1)
#y2 = shuffle(y1, random_state=1)
#y3 = shuffle(y1, random_state=2)
#Y = np.vstack((y1, y2, y3)).T

#n_samples, n_features = X.shape # 10,100
#n_outputs = Y.shape[1] # 3
#n_classes = 3
#forest = RandomForestClassifier(n_estimators=100, random_state=1)
#multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
#multi_target_forest.fit(X, Y)

# WITH DATA MADE BEFORE 
n_samples, n_features = X_train.shape # 10,100
#n_outputs = y.shape[1] # 3
#n_classes = 3
forest = RandomForestClassifier(n_estimators=100, random_state=1)
multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
multi_target_forest.fit(X_train, y_train)

predictions = multi_target_forest.predict(X_test)
from sklearn.metrics import accuracy_score , classification_report
print('accuracy:',  accuracy_score(y_test,predictions))
print(classification_report(y_test, predictions))








