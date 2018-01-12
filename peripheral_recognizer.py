# EMG, and other peripheral signal recognition of affect
import numpy as np
import signalpreprocess as sp
import cPickle
import math

# For the SVM
import pandas as pd
from sklearn import datasets
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, cross_validate
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
X, y = cPickle.load(open('deap_data/peripheral_data.dat', 'rb'))
X = np.array(X)
y = np.array(y)

y_valence = np.array(sp.data_binarizer([el[0] for el in y],5))
y_arousal = np.array(sp.data_binarizer([el[1] for el in y],5))
y_dominance = np.array(sp.data_binarizer([el[2] for el in y],5))

# Describe the data
valencepd = pd.Categorical(y_valence)
arousalpd = pd.Categorical(y_arousal)
dominancepd = pd.Categorical(y_dominance)
print (valencepd.describe())
print (arousalpd.describe())
print (dominancepd.describe())

# pca_result = SelectKBest(f_classif, k=5).fit_transform(X, y_valence)
pca_result = X

####################
##### Valence ######
####################
# print ("This is Valence")
# g_range = 2. ** np.arange(-15,5,step = 1)
# C_range = 2. ** np.arange(-15,5,step=1)
# parameters = [{'gamma':g_range, 'C':C_range, 'kernel':['rbf']}]
# grid = GridSearchCV(svm.SVC(), parameters, cv = 10, n_jobs = -1)
# grid.fit(X, y_valence)
# bestG = grid.best_params_['gamma']
# bestC = grid.best_params_['C']
# print("The best parameters for valence are: gamma=", np.log2(bestG), " and Cost = ", np.log2(bestC))
# SVM_rbf_valence = svm.SVC(kernel='rbf', C= (bestC), gamma= (bestG)) # For RBF Kernel

SVM_rbf_valence = svm.SVC()
print "F1 SCORES \n"
a = cross_val_score(SVM_rbf_valence, pca_result, y_valence, cv= 10, scoring = 'f1', n_jobs=-1)
print ((np.mean(a)), (np.std(a)))

print "\n ACCURACY SCORES \n"
# Create and fit the Model using the training data
a = cross_val_score(SVM_rbf_valence, pca_result, y_valence, cv= 10, scoring = 'accuracy', n_jobs=-1)
print ((np.mean(a)), (np.std(a)))



####################
##### Arousal ######
####################
# print ("This is Arousal")
# g_range = 2. ** np.arange(-15,5,step =1)
# C_range = 2. ** np.arange(-15,5,step=1)
# parameters = [{'gamma':g_range, 'C':C_range, 'kernel':['rbf']}]
# grid = GridSearchCV(svm.SVC(), parameters, cv = 10, n_jobs = -1)
# grid.fit(X, y_valence)
# bestG = grid.best_params_['gamma']
# bestC = grid.best_params_['C']
# print("The best parameters for valence are: gamma=", np.log2(bestG), " and Cost = ", np.log2(bestC))
# SVM_rbf_arousal = svm.SVC(kernel='rbf', C= (bestC), gamma= (bestG)) # For RBF Kernel

# a = cross_val_score(SVM_rbf_arousal, pca_result, y_arousal, cv= 32, scoring = 'f1', n_jobs=-1)
# print ((np.mean(a)), (np.std(a)))


# a = cross_val_score(SVM_rbf_arousal, pca_result, y_arousal, cv= 32, scoring = 'accuracy', n_jobs=-1)
# print ((np.mean(a)), (np.std(a)))


####################
##### Dominance ####
####################
# print ("This is Dominance")
# g_range = 2. ** np.arange(-15,5,step = 1)
# C_range = 2. ** np.arange(-15,5,step=1)
# parameters = [{'gamma':g_range, 'C':C_range, 'kernel':['rbf']}]
# grid = GridSearchCV(svm.SVC(), parameters, cv = 10, n_jobs = -1)
# grid.fit(X, y_valence)
# bestG = grid.best_params_['gamma']
# bestC = grid.best_params_['C']
# print("The best parameters for valence are: gamma=", np.log2(bestG), " and Cost = ", np.log2(bestC))
# SVM_rbf_dominance = svm.SVC(kernel='rbf', C= (bestC), gamma= (bestG)) # For RBF Kernel

# a = cross_val_score(SVM_rbf_dominance, pca_result, y_dominance, cv= 32, scoring = 'f1', n_jobs=-1)
# print ((np.mean(a)), (np.std(a)))

# a = cross_val_score(SVM_rbf_dominance, pca_result, y_dominance, cv= 32, scoring = 'accuracy', n_jobs=-1)
# print ((np.mean(a)), (np.std(a)))
