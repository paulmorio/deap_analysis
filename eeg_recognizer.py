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
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
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

# Describe the data
valencepd = pd.Categorical(y_valence)
arousalpd = pd.Categorical(y_arousal)
dominancepd = pd.Categorical(y_dominance)
print (valencepd.describe())
print (arousalpd.describe())
print (dominancepd.describe())


# pca = PCA(n_components=5)
# pca_result = pca.fit_transform(X)
# print 'Explained variation per principal component: {}'.format(pca.explained_variance_ratio_)

pca_result = X
####################
##### Valence ######
####################

# Create and fit the Model using the training data
gnb = svm.SVC()
print "F1 SCORES \n"
a = cross_val_score(gnb, pca_result, y_valence, cv= 32, scoring = 'f1', n_jobs=-1)
print ((np.mean(a)), (np.std(a)))

a = cross_val_score(gnb, pca_result, y_arousal, cv= 32, scoring = 'f1', n_jobs=-1)
print ((np.mean(a)), (np.std(a)))

a = cross_val_score(gnb, pca_result, y_dominance, cv= 32, scoring = 'f1', n_jobs=-1)
print ((np.mean(a)), (np.std(a)))

print "\n ACCURACY SCORES \n"
# Create and fit the Model using the training data
a = cross_val_score(gnb, pca_result, y_valence, cv= 32, scoring = 'accuracy', n_jobs=-1)
print ((np.mean(a)), (np.std(a)))

a = cross_val_score(gnb, pca_result, y_arousal, cv= 32, scoring = 'accuracy', n_jobs=-1)
print ((np.mean(a)), (np.std(a)))

a = cross_val_score(gnb, pca_result, y_dominance, cv= 32, scoring = 'accuracy', n_jobs=-1)
print ((np.mean(a)), (np.std(a)))

