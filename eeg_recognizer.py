# Affect recognition via EEG
import numpy as np
import signalpreprocess as sp
import eeg_features as ef
import cPickle

# For the SVM
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib as plt
from sklearn import metrics
from sklearn import tree

# # Load the entire 32 patients accesible by number.
# raw_data_dict = cPickle.load(open('deap_data/data_preprocessed_python/all_32.dat', 'rb'))

# ## Need to do this for all 32 and first 22 seperately to make sure we are doing things properly.

###################################################################################
############################ Full 32 Participants #################################
###################################################################################
# # each participant has their data styled in the following pattern (in a dict)
# # 'data'	40 x 40 x 8064	video/trial x channel x data
# # 'labels'	40 x 4	video/trial x label (valence, arousal, dominance, liking)

# participants = range(1,32)
# videos = range(1,40)
# channels = range(1,33) # we are interested only in the EEG data at the moment. 

# # Our data for the ml
# X = [] # make full dumb
# y = [] # put all ratings, so we can subset laters

# # construct feature vectors
# for person in participants:
# 	for vid in videos:
# 		channels_data = ((raw_data_dict[person])['data'])[vid]
# 		ratings = ((raw_data_dict[person]['labels'])[vid])
# 		y.append(ratings) # append video ratingS to labels

# 		# Our data vector
# 		x = []

# 		# The left right assymetry signals
# 		lr_pfl = ef.lr_assymetry_pfl(channels_data)
# 		lr_ears = ef.lr_assymetry_ears(channels_data)
# 		lr_back = ef.lr_assymetry_back(channels_data)

# 		# feature extraction of the assymetry
# 		lr_pfl_m, lr_pfl_s, lr_pfl_nfd, lr_pfl_nsd = sp.package_deal_signal(lr_pfl)
# 		lr_ears_m, lr_ears_s, lr_ears_nfd, lr_ears_nsd = sp.package_deal_signal(lr_ears)
# 		lr_back_m, lr_back_s, lr_back_nfd, lr_back_nsd = sp.package_deal_signal(lr_back)

# 		x = x + [lr_pfl_m, lr_pfl_s, lr_pfl_nfd, lr_pfl_nsd, lr_ears_m, lr_ears_s, 
# 				lr_ears_nfd, lr_ears_nsd, lr_back_m, lr_back_s, lr_back_nfd, lr_back_nsd]
		
# 		# eeg_w for all the channels
# 		eeg_w_list = []
# 		for signal in channels_data:
# 			eeg_w_list.append(ef.eeg_w_beta(signal))
# 		x = x + eeg_w_list

# 		X.append(x)
# 	print person

# y_valence = sp.data_binarizer([el[0] for el in y],5)
# y_arousal = sp.data_binarizer([el[1] for el in y],5)
# y_dominance = sp.data_binarizer([el[2] for el in y],5)

###################################################################################
############################ Full 32 Participants #################################
###################################################################################
# Data loader edition
X, y = cPickle.load(open('deap_data/eeg_data.dat', 'rb'))
y_valence = sp.data_binarizer([el[0] for el in y],5)
y_arousal = sp.data_binarizer([el[1] for el in y],5)
y_dominance = sp.data_binarizer([el[2] for el in y],5)

####################
##### Valence ######
####################

test_size = 200
X_train = X[:-test_size]
y_train = y_valence[:-test_size]
X_test = X[-test_size:]
y_test = y_valence[-test_size:]

# Create and fit the Model using the training data
SVM_linear = svm.SVC(kernel= 'linear', C=1)
SVM_linear.fit(X_train, y_train)

# Do some Metrics
from sklearn import metrics
y_pred = SVM_linear.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
print("Overall Accuracy:", round(metrics.accuracy_score(y_test, y_pred),2))



# Crate and fit a decision tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
print("Overall Accuracy:", round(metrics.accuracy_score(y_test, y_pred),2))


#######
# RBF CV
# RBF
g_range = 2. ** np.arange(-10,10,step = 1)
C_range = 2. ** np.arange(-10,10,step=1)
parameters = [{'gamma':g_range, 'C':C_range, 'kernel':['rbf']}]
grid = GridSearchCV(svm.SVC(), parameters, cv = 10, n_jobs = 4)
grid.fit(X_train, y_train)
bestG = grid.best_params_['gamma']
bestC = grid.best_params_['C']
print("The best parameters are: gamma=", np.log2(bestG), " and Cost = ", np.log2(bestC))

SVM_rbf = svm.SVC(kernel='rbf', C= (bestC), gamma= (bestG)) # For RBF Kernel
SVM_rbf.fit(X_train, y_train)

# RBF metrics
y_pred = SVM_rbf.predict(X_test)
print("~~~~RBF~~~~")
print(metrics.classification_report(y_test, y_pred))
print("Overall Accuracy:", round(metrics.accuracy_score(y_test, y_pred),2))

###################################################################################
########################### First 22 Participants #################################
###################################################################################

