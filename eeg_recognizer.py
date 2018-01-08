# Affect recognition via EEG

import signalpreprocess as sp
import cPickle


# Load the data and run the preprocessing bits

# Load the entire 32 patients accesible by number.
raw_data_dict = cPickle.load(open('deap_data/data_preprocessed_python/all_32.dat', 'rb'))
print type(raw_data_dict)


## Need to do this for all 32 and first 22 seperately to make sure we are doing things properly.