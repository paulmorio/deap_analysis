# DEAP preprocessed data construction

# Lets get a brief overview of one piece of data
import cPickle
x = cPickle.load(open('deap_data/data_preprocessed_python/s01.dat', 'rb'))
print type(x)
print x['labels'].shape
print x['data'].shape

# Array name	Array shape	Array contents
# data	40 x 40 x 8064	video/trial x channel x data
# labels	40 x 4	video/trial x label (valence, arousal, dominance, liking)

# 1. The data was downsampled to 128Hz.
# 2. EOG artefacts were removed as in [1].
# 3. A bandpass frequency filter from 4.0-45.0Hz was applied.
# 4. The data was averaged to the common reference.
# The EEG channels were reordered so that they all follow the Geneva order as above.
# The data was segmented into 60 second trials and a 3 second pre-trial baseline removed.
# The trials were reordered from presentation order to video (Experiment_id) order.
