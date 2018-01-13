# Combined data, single trial classification
# Requires existence of the EEG and the peripheral signal data

import numpy as np
import signalpreprocess as sp
import peripheral_features as pf
import cPickle
import sys
import glob
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('imgs', type=str, nargs='+', help="Input image representations per participant.")
args = parser.parse_args()

# load the data from the relevant modalities (except the image which requires the table merging)
# EEG
st_X_eeg, st_y_eeg = cPickle.load(open('deap_data/eeg_data.dat', 'rb'))
# Peripheral
st_X_per, st_y_per = cPickle.load(open('deap_data/peripheral_data.dat', 'rb'))

# Multimodal
st_X = []
st_y = []

# import and look at the type and shape of the participant_ratings file
participant_data = pd.read_csv("deap_data/metadata_csv/participant_ratings.csv")
person_count = 1
for person in args.imgs:
	print person_count
	X = []
	y = []
	embeddingsDict = cPickle.load(open(person, 'rb'))
	for key, imageRep in sorted(embeddingsDict.iteritems()):
		# Make the general entry ratings, and keep note of the person count (for the other data sets)
		a = participant_data[participant_data['Participant_id'] == person_count]
		b = participant_data[participant_data['Trial'] == key]
		partRating = pd.merge(a,b)
		y_entry = [np.float(partRating.Valence), np.float(partRating.Arousal),
					np.float(partRating.Dominance), np.float(partRating.Liking)] #val,arousal,dom,liking
		y.append(y_entry)

		# Create the X_entry
		experiment_id = np.int(partRating.Experiment_id)-1
		X_entry = []

		# image data
		X_entry.extend(imageRep)

		# EEG Data
		print len(st_X_eeg)
		print person_count
		print experiment_id
		eeg_x = (st_X_eeg[(person_count-1)])[experiment_id]
		X_entry.extend(eeg_x)
		
		# Peripheral Data
		peri_x = (st_X_per[(person_count-1)])[experiment_id]
		X_entry.extend(peri_x)

		X.append(X_entry)
	
	st_X.append(X)
	st_y.append(y)
	person_count += 1

f = open('deap_data/multimodal_data.dat', 'wb')
cPickle.dump((st_X,st_y), f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()