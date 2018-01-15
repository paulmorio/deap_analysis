# Affect recognition via EEG
import numpy as np
import signalpreprocess as sp
import eeg_features as ef
import cPickle
import math

# Machinelearning
import pandas as pd
from sklearn import datasets
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib as plt
from sklearn import metrics
from sklearn import tree

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, f_regression

###################################################################################
############################### Load Dataset ######################################
###################################################################################

# Data loader edition
X, y = cPickle.load(open('deap_data/eeg_data_old.dat', 'rb'))
X = np.array(X)
y = np.array(y)

y_valence = np.array(sp.data_binarizer([el[0] for el in y],5))
y_arousal = np.array(sp.data_binarizer([el[1] for el in y],5))
y_dominance = np.array(sp.data_binarizer([el[2] for el in y],5))
y_liking = np.array(sp.data_binarizer([el[3] for el in y],5))


# Describe the data
arousalpd = pd.Categorical(y_arousal)
valencepd = pd.Categorical(y_valence)
dominancepd = pd.Categorical(y_dominance)
likingpd = pd.Categorical(y_liking)
print (arousalpd.describe())
print (valencepd.describe())
print (dominancepd.describe())
print (likingpd.describe())

# # pca_result = SelectKBest(f_classif, k=5).fit_transform(X, y_valence)
# pca_result = X

# ####################
# ##### Valence ######
# ####################
# # print ("This is Valence")

# SVM_rbf_valence = svm.SVC()
# print "F1 SCORES \n"
# a = cross_val_score(SVM_rbf_valence, pca_result, y_valence, cv= 5, scoring = 'f1', n_jobs=-1)
# print ((np.mean(a)), (np.std(a)))

# print "\n ACCURACY SCORES \n"
# # Create and fit the Model using the training data
# a = cross_val_score(SVM_rbf_valence, pca_result, y_valence, cv= 5, scoring = 'accuracy', n_jobs=-1)
# print ((np.mean(a)), (np.std(a)))

####################
##### Arousal ######
####################
# print ("This is Arousal")

# a = cross_val_score(SVM_rbf_arousal, pca_result, y_arousal, cv= 32, scoring = 'f1', n_jobs=-1)
# print ((np.mean(a)), (np.std(a)))


# a = cross_val_score(SVM_rbf_arousal, pca_result, y_arousal, cv= 32, scoring = 'accuracy', n_jobs=-1)
# print ((np.mean(a)), (np.std(a)))


####################
##### Dominance ####
####################
# print ("This is Dominance")

# a = cross_val_score(SVM_rbf_dominance, pca_result, y_dominance, cv= 32, scoring = 'f1', n_jobs=-1)
# print ((np.mean(a)), (np.std(a)))

# a = cross_val_score(SVM_rbf_dominance, pca_result, y_dominance, cv= 32, scoring = 'accuracy', n_jobs=-1)
# print ((np.mean(a)), (np.std(a)))
