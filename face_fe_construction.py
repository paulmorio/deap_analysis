# facial embedding fe construction

# Affect recognition via EEG
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
	for key, value in sorted(embeddingsDict.iteritems()):
		a = participant_data[participant_data['Participant_id'] == person_count]
		b = participant_data[participant_data['Trial'] == key]
		partRating = pd.merge(a,b)
		y_entry = [np.float(partRating.Valence), np.float(partRating.Arousal),
					np.float(partRating.Dominance), np.float(partRating.Liking)] #val,arousal,dom,liking
		y.append(y_entry)
		X.append(value)
	
	st_X.append(X)
	st_y.append(y)
	person_count += 1

f = open('deap_data/facerep_data.dat', 'wb')
cPickle.dump((st_X,st_y), f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()