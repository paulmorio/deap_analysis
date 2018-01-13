# single trial peripheral recognizer
# EMG, and other peripheral signal recognition of affect
import numpy as np
import signalpreprocess as sp
import cPickle
import math

# For the SVM
import pandas as pd
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn import metrics
from sklearn import tree

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, f_regression

###################################################################################
################### Classification Functions ######################################
###################################################################################

def looCV(X,y):
	# since scikit learn doesnt implement this beauty
	loo = LeaveOneOut()
	classification_accuracy = []
	f1score = []
	for train_index, test_index in loo.split(X):
		train_X, test_X = X[train_index], X[test_index]
		train_y, test_y = y[train_index], y[test_index]

		# evilknievel edition
		g_range = 2. ** np.arange(-15,5,step = 1)
		C_range = 2. ** np.arange(-15,5,step=1)
		parameters = [{'gamma':g_range, 'C':C_range, 'kernel':['rbf']}]
		grid = GridSearchCV(svm.SVC(), parameters, cv = 3, n_jobs = -1)
		grid.fit(train_X, train_y)
		bestG = grid.best_params_['gamma']
		bestC = grid.best_params_['C']
		print("The best parameters for valence are: gamma=", np.log2(bestG), " and Cost = ", np.log2(bestC))
		svmRBF = svm.SVC(kernel='rbf', C= (bestC), gamma= (bestG)) # For RBF Kernel
		svmRBF.fit(train_X, train_y)

		# svmRBF = svm.SVC()
		# svmRBF.fit(train_X, train_y)
		y_pred = svmRBF.predict(test_X)
		classification_accuracy.append(metrics.accuracy_score(test_y, y_pred))
		f1score.append(metrics.f1_score(test_y, y_pred))

	return np.mean(classification_accuracy), np.mean(f1score)


###################################################################################
############################### Load Dataset ######################################
###################################################################################
st_X, st_y = cPickle.load(open('deap_data/peripheral_data.dat', 'rb'))

classification_accuracy_valence = []
f1score_valence = []

classification_accuracy_arousal = []
f1score_arousal = []

classification_accuracy_dominance = []
f1score_dominance = []

classification_accuracy_liking = []
f1score_liking = []

count = 0
for X,y in zip(st_X, st_y):
	print count
	count +=1
	# Data loader edition
	X = np.array(X)
	y = np.array(y)

	y_valence = np.array(sp.data_binarizer([el[0] for el in y],5))
	y_arousal = np.array(sp.data_binarizer([el[1] for el in y],5))
	y_dominance = np.array(sp.data_binarizer([el[2] for el in y],5))
	y_liking = np.array(sp.data_binarizer([el[3] for el in y],5))

	# Describe the data
	# valencepd = pd.Categorical(y_valence)
	# arousalpd = pd.Categorical(y_arousal)
	# dominancepd = pd.Categorical(y_dominance)
	# print (valencepd.describe())
	# print (arousalpd.describe())
	# print (dominancepd.describe())

	# pca_result = SelectKBest(f_classif, k=5).fit_transform(X, y_valence)
	pca_result = X

	####################
	##### Valence ######
	####################
	ca, f1 =  (looCV(pca_result,y_valence))
	classification_accuracy_valence.append(ca)
	f1score_valence.append(f1)

	####################
	##### Arousal ######
	####################
	ca, f1 =  (looCV(pca_result,y_arousal))
	classification_accuracy_arousal.append(ca)
	f1score_arousal.append(f1)

	####################
	##### Dominance ####
	####################
	if count == 27:
		classification_accuracy_dominance.append(1.0)
		f1score_dominance.append(1.0)
	else:
		ca, f1 =  (looCV(pca_result,y_dominance))
		classification_accuracy_dominance.append(ca)
		f1score_dominance.append(f1)

	####################
	##### Liking #######
	####################
	ca, f1 =  (looCV(pca_result,y_liking))
	classification_accuracy_liking.append(ca)
	f1score_liking.append(f1)


print ("This is Valence")
print np.mean(classification_accuracy_valence), np.mean(f1score_valence)

print ("This is Arousal")
print np.mean(classification_accuracy_arousal), np.mean(f1score_arousal)

print ("This is Dominance")
print np.mean(classification_accuracy_dominance), np.mean(f1score_dominance)

print ("This is Liking")
print np.mean(classification_accuracy_liking), np.mean(f1score_liking)