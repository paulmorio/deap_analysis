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

participants = range(1,32)
videos = range(1,40)
channels = range(1,33) # we are interested only in the EEG data at the moment. 

# Our data for the ml
X = [] # make full dumb
y = [] # put all ratings, so we can subset laters

# construct feature vectors
for person in participants:
	for vid in videos:
		channels_data = ((raw_data_dict[person])['data'])[vid]
		ratings = ((raw_data_dict[person]['labels'])[vid])
		y.append(ratings) # append video ratingS to labels

		# Our data vector
		x = []

		# The left right assymetry signals
		lr_pfl = ef.lr_assymetry_pfl(channels_data)
		lr_ears = ef.lr_assymetry_ears(channels_data)
		lr_back = ef.lr_assymetry_back(channels_data)

		# feature extraction of the assymetry
		lr_pfl_m, lr_pfl_s, lr_pfl_nfd, lr_pfl_nsd = sp.package_deal_signal(lr_pfl)
		lr_ears_m, lr_ears_s, lr_ears_nfd, lr_ears_nsd = sp.package_deal_signal(lr_ears)
		lr_back_m, lr_back_s, lr_back_nfd, lr_back_nsd = sp.package_deal_signal(lr_back)

		x = x + [lr_pfl_m, lr_pfl_s, lr_pfl_nfd, lr_pfl_nsd, lr_ears_m, lr_ears_s, 
				lr_ears_nfd, lr_ears_nsd, lr_back_m, lr_back_s, lr_back_nfd, lr_back_nsd]
		
		# # eeg_w for all the channels
		# eeg_w_list = []
		# for signal in channels_data:
		# 	eeg_w_list.append(ef.eeg_w_beta(signal))
		# x = x + eeg_w_list

		X.append(x)
	print person

f = open('deap_data/eeg_data.dat', 'wb')
cPickle.dump((X,y), f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()