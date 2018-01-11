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
		channels_data = (((raw_data_dict[person])['data'])[vid])[:32]
		ratings = ((raw_data_dict[person]['labels'])[vid])
		y.append(ratings) # append video ratingS to labels

		# Our data vector
		x = []

		# # The left right assymetry signals alpha
		# lr_pfl = ef.lr_assymetry_pfl(channels_data)
		# lr_ears = ef.lr_assymetry_ears(channels_data)
		# lr_back = ef.lr_assymetry_back(channels_data)
		# x = x + lr_pfl.tolist() + lr_ears.tolist() + lr_back.tolist()

		# # The left right assymetry signals theta
		# lr_pfl = ef.lr_assymetry_pfl_t(channels_data)
		# lr_ears = ef.lr_assymetry_ears_t(channels_data)
		# lr_back = ef.lr_assymetry_back_t(channels_data)
		# x = x + lr_pfl.tolist() + lr_ears.tolist() + lr_back.tolist()

		# energy log sum ratios eeg_w for all the channels
		eeg_w_list = []
		for signal in channels_data:
			eeg_w_list.append(ef.eeg_w_beta(signal))
		x = x + eeg_w_list

		# # powerspectra for all the channels
		# powerspectra = []
		# for signal in channels_data:
		# 	powerspectra.extend(ef.power_spectrums(signal))
		# x = x + powerspectra

		# # simple stats
		# sstats = []
		# for signal in channels_data:
		# 	m,s,nfd,nsd, mini, maxi = sp.package_deal_signal_mm(signal)
		# 	sstats += [m,s,nfd,nsd, mini, maxi]
		# x = x + sstats


		X.append(x)
	print person

f = open('deap_data/eeg_data.dat', 'wb')
cPickle.dump((X,y), f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()