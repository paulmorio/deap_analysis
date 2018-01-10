# Affect recognition via EEG
import numpy as np
import signalpreprocess as sp
import eeg_features as ef
import cPickle
import math

# For the SVM
from sklearn import datasets
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib as plt
from sklearn import metrics
from sklearn import tree


###################################################################################
############################### Load Dataset ######################################
###################################################################################

# Data loader edition
X, y = cPickle.load(open('deap_data/eeg_data.dat', 'rb'))
y_valence = np.array(sp.data_binarizer([el[0] for el in y],5))
y_arousal = np.array(sp.data_binarizer([el[1] for el in y],5))
y_dominance = np.array(sp.data_binarizer([el[2] for el in y],5))

X = np.array(X)
y = np.array(y)
####################
##### Valence ######
####################

indices = np.random.permutation(len(X))
test_size = int(math.ceil(len(X)/32))
print test_size
print indices
X_train = X[indices[:-test_size]]
y_train = y_valence[indices[:-test_size]]
X_test = X[indices[-test_size:]]
y_test = y_valence[indices[-test_size:]]

# Create and fit the Model using the training data
SVM_linear = svm.SVC(kernel= 'rbf', C=0.01)
SVM_linear.fit(X_train, y_train)

# Do some Metrics
from sklearn import metrics
y_pred = SVM_linear.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
print("Overall Accuracy:", round(metrics.accuracy_score(y_test, y_pred),2))






# Create and fit the Model using the training data
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Do some Metrics
from sklearn import metrics
y_pred = gnb.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
print("Overall Accuracy:", round(metrics.accuracy_score(y_test, y_pred),2))


# Create and fit the Model using the training data
gnb = DecisionTreeClassifier(max_depth=5)
gnb.fit(X_train, y_train)

# Do some Metrics
from sklearn import metrics
y_pred = gnb.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
print("Overall Accuracy:", round(metrics.accuracy_score(y_test, y_pred),2))