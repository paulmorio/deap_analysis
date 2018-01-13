# single trial eeg classifier edition
# EEG data constructor and feature extraction file
# In this file we traverse the data in EEG, perform features extraction and save the data cause its big and takes time.

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

participants = range(1,33)
videos = range(0,40)
channels = range(1,33) # we are interested only in the EEG data at the moment. 

print len((raw_data_dict[1])['data'])

# Our data for the ml
st_X = []
st_y = []

# construct feature vectors
for person in participants:
	X = [] # make full dumb
	y = [] # put all ratings, so we can subset laters
	for vid in videos:
		channels_data = (((raw_data_dict[person])['data'])[vid])[:32]
		ratings = ((raw_data_dict[person]['labels'])[vid])
		y.append(ratings) # append video ratingS to labels

		# Our data vector
		x = []

		# The left right assymetry signals alpha
		lr_pfl = ef.lr_assymetry_pfl(channels_data, "alpha")
		lr_ears = ef.lr_assymetry_ears(channels_data, "alpha")
		lr_back = ef.lr_assymetry_back(channels_data, "alpha")
		x = x + [lr_pfl, lr_ears, lr_back]

		# The left right assymetry signals theta
		lr_pfl = ef.lr_assymetry_pfl(channels_data, "theta")
		lr_ears = ef.lr_assymetry_ears(channels_data, "theta")
		lr_back = ef.lr_assymetry_back(channels_data, "theta")
		x = x + [lr_pfl, lr_ears, lr_back]

		# The left right assymetry signals theta
		lr_pfl = ef.lr_assymetry_pfl(channels_data, "beta")
		lr_ears = ef.lr_assymetry_ears(channels_data, "beta")
		lr_back = ef.lr_assymetry_back(channels_data, "beta")
		x = x + [lr_pfl, lr_ears, lr_back]

		# energy log sum ratios eeg_w for all the channels
		eeg_w_list = []
		for signal in channels_data:
			eeg_w_list.append(ef.eeg_w_beta(signal))
		x = x + eeg_w_list

		# powerspectrumstuff
		for signal in channels_data:
			x = x + ef.power_spectrums(signal)


		X.append(x)
	print person
	st_X.append(X)
	st_y.append(y)

f = open('deap_data/eeg_data.dat', 'wb')
cPickle.dump((st_X,st_y), f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()