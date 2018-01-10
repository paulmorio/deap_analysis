# EMG and peripheral data feature extraction and construction. 

# Affect recognition via EEG
import numpy as np
import signalpreprocess as sp
import eeg_features as ef
import cPickle


# Load the entire 32 patients accesible by number.
raw_data_dict = cPickle.load(open('deap_data/data_preprocessed_python/all_32.dat', 'rb'))

## Need to do this for all 32 and first 22 seperately to make sure we are doing things properly.

###################################################################################
############################ Full 32 Participants #################################
###################################################################################
# each participant has their data styled in the following pattern (in a dict)
# 'data'	40 x 40 x 8064	video/trial x channel x data
# 'labels'	40 x 4	video/trial x label (valence, arousal, dominance, liking)

participants = range(1,3)
videos = range(1,40)
channels = range(33,41) # we are interested only in the EEG data at the moment. 

# Our data for the ml
X = [] # make full dumb
y = [] # put all ratings, so we can subset laters

# construct feature vectors
for person in participants:
	for vid in videos:
		channels_data = ((raw_data_dict[person])['data'])[vid]
		ratings = ((raw_data_dict[person]['labels'])[vid])
		y.append(ratings) # append video ratingS to labels

		print person
		print channels_data
		# Our data vector
		x = []

		X.append(x)
	print person

f = open('deap_data/peripheral_data.dat', 'wb')
cPickle.dump((X,y), f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()