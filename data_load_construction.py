# DEAP preprocessed data construction

# Lets get a brief overview of one piece of data
import cPickle
x = cPickle.load(open('deap_data/data_preprocessed_python/s01.dat', 'rb'))
print type(x)
print x['labels'].shape
print x['data'].shape
