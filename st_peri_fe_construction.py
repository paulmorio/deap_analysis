# single trial peripheral features data constructor
# EMG and peripheral data feature extraction and construction. 

# Affect recognition via EEG
import numpy as np
import signalpreprocess as sp
import peripheral_features as pf
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

participants = range(1,33)
videos = range(0,40)

# Our data for the ml
st_X = []
st_y = []

# construct feature vectors
for person in participants:
	X = [] # make full dumb
	y = [] # put all ratings, so we can subset laters
	for vid in videos:
		channels_data = (((raw_data_dict[person])['data'])[vid])[32:]
		ratings = ((raw_data_dict[person]['labels'])[vid])
		y.append(ratings) # append video ratingS to labels

		# Our data vector
		x = []

		# Add features to our feature vector as necessary
		for signal in channels_data:
			m, s, nfd, nsd, mini, maxi = sp.package_deal_signal_mm(signal)
			x.extend([m,s,nfd,nsd, mini, maxi])
			x.append(pf.power_spectrums(signal))
			x.append(pf.average_gradient(signal))

		X.append(x)
	print person
	st_X.append(X)
	st_y.append(y)

f = open('deap_data/peripheral_data.dat', 'wb')
cPickle.dump((st_X,st_y), f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()