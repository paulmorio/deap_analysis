# Affect recognition via EEG

import signalpreprocess as sp
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
videos = range(1,40)
channels = range(1,33) # we are interested only in the EEG data at the moment. 
# Maybe we just want certain smart features

# construct feature vector

raw_data_dict[1]['data'][1]

###################################################################################
########################### First 22 Participants #################################
###################################################################################
