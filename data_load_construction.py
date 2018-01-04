# DEAP preprocessed data construction
# Lets get a brief overview of one piece of data

import cPickle
x = cPickle.load(open('deap_data/data_preprocessed_python/s01.dat', 'rb'))
print type(x)
print x['labels'].shape
print x['data'].shape

data = []
for i in range(1,33):
	if i < 10:
		expId = "0" + str(i)
	else:
		expId = str(i)

	file_to_load = "deap_data/data_preprocessed_python/s" + expId + ".dat"
	par_experiment_data = cPickle.load(open(file_to_load, 'rb'))
	print(expId)
	data.append(par_experiment_data)

f = open('deap_data/data_preprocessed_python/all_32.dat', 'wb')
cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()
