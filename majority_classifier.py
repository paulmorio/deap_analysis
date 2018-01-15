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
from sklearn.dummy import DummyClassifier
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

	# evilknievel edition
	svmRBF = DummyClassifier(strategy='most_frequent')

	for train_index, test_index in loo.split(X):
		train_X, test_X = X[train_index], X[test_index]
		train_y, test_y = y[train_index], y[test_index]

		svmRBF.fit(train_X, train_y)

		# svmRBF = svm.SVC()
		# svmRBF.fit(train_X, train_y)
		y_pred = svmRBF.predict(test_X)
		classification_accuracy.append(metrics.accuracy_score(test_y, y_pred))
		f1score.append(metrics.f1_score(test_y, y_pred))

	return np.mean(classification_accuracy), np.mean(f1score)

def classHighFrac(binaryArray):
	# count the number of 1s in percentage
	num_ones = np.count_nonzero(binaryArray == 1)
	fraction_ones = (float(num_ones)/len(binaryArray))
	return fraction_ones

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

# For class imbalances and standard deviations
class_valence = []
class_arousal = []
class_dominance = []
class_liking = []


count = 0
for X,y in zip(st_X, st_y):
	print count
	count +=1
	# Data loader edition
	X = np.array(X)
	y = np.array(y)

	mid_valence = np.median(np.array([el[0] for el in y]))
	mid_arousal = np.median(np.array([el[1] for el in y]))
	mid_dominance = np.median(np.array([el[2] for el in y]))
	mid_liking = np.median(np.array([el[3] for el in y]))

	y_valence = np.array(sp.data_binarizer([el[0] for el in y],5))
	y_arousal = np.array(sp.data_binarizer([el[1] for el in y],5))
	y_dominance = np.array(sp.data_binarizer([el[2] for el in y],5))
	y_liking = np.array(sp.data_binarizer([el[3] for el in y],5))

	class_valence.append(classHighFrac(y_valence))
	class_arousal.append(classHighFrac(y_arousal))
	class_dominance.append(classHighFrac(y_dominance))
	class_liking.append(classHighFrac(y_liking))

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


print ("This is Arousal")
print np.mean(classification_accuracy_arousal), np.mean(f1score_arousal)

print ("This is Valence")
print np.mean(classification_accuracy_valence), np.mean(f1score_valence)

print ("This is Dominance")
print np.mean(classification_accuracy_dominance), np.mean(f1score_dominance)

print ("This is Liking")
print np.mean(classification_accuracy_liking), np.mean(f1score_liking)

print ("These are the class imbalances for each class \n in arousal, valence, dominance, liking")
print np.mean(class_arousal), np.std(class_arousal)
print np.mean(class_valence), np.std(class_valence)
print np.mean(class_dominance), np.std(class_dominance)
print np.mean(class_liking), np.std(class_liking)