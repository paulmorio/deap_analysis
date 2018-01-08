# DEAP preprocessed data construction
# Lets get a brief overview of one piece of data

import cPickle
import pandas as pd
import numpy as np 

# import and look at the type and shape of the participant_ratings file
participant_data = pd.read_csv("deap_data/metadata_csv/participant_ratings.csv")

# Import the raw unmatched and randomized image and dictionarise by 
# their participant id (which in turn contains another dictionary)
data = {}
for i in range(1,22):
	if i < 10:
		participantId = "0" + str(i)
	else:
		participantId = str(i)

	file_to_load = "deap_data/img_representations/s" + participantId + ".dat"
	par_experiment_data = cPickle.load(open(file_to_load, 'rb'))
	data[participantId] = par_experiment_data

embedding_data = {}

for i in range(1,22):
	# i is the participant id
	rep_dict = data[i]
	embedding_data[i] = {}
	for trial in range(1,40):
		if trial in rep_dict:
			# facial features exist hence we can match up crap
			relevant_row = (participant_data.loc[participant_data['Participant_id'] == i & (participant_data['Trial'] == trial)])
			expid = relevant_row['Experiment_id']
			arousal = relevant_row['Arousal']
			valence = relevant_row['Valence']
			dominance = relevant_row['Dominance']
			face_embedding = rep_dict[trial]

			(embedding_data[i])[expid] = [face_embedding, arousal, valence, dominance]
			print (embedding_data[i])[expid]

f = open('deap_data/img_representations/all_32_img.dat', 'wb')
cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()
