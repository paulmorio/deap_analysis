# This module is dedicated to summarizing single channel signal data and extracting features from it.

import numpy as np

def normalise_signal(signal):
	"""Normalise the signal to zero mean, unit variance"""
	signal_mean = np.mean(signal)
	signal_stdev = np.std(signal)
	signal_normed = [((x-signal_mean)/signal_stdev) for x in signal]
	return(signal_normed)

def mean(signal):
	""" The mean of the raw signal"""
	return(np.mean(signal))

def std_dev(signal):
	"""The standard deviation of the raw signal"""
	return(np.std(signal))

def first_differences(signal):
	"""The mean of the absolute values of the first differences of the raw signal"""
	# build the first differences list
	first_diff = []
	for i in range(0,len(signal)-1):
		first_diff.append(abs(signal[i+1]-signal[i]))

	fd_sum = sum(first_diff)
	delta = float(fd_sum)/(len(signal)-1)
	return(delta)

def norm_first_differences(signal):
	"""The mean of the absolute values of the first differences of the normalized signal"""
	signal_normed = normalise_signal(signal)
	first_diff = []
	for i in range(0,len(signal_normed)-1):
		first_diff.append(abs(signal_normed[i+1]-signal_normed[i]))

	fd_sum = sum(first_diff)
	delta = float(fd_sum)/(len(signal_normed)-1)
	return(delta)

def second_differences(signal):
	"""The mean of the absolute values of the second differences of the raw signal"""
	sec_diff = []
	for i in range(0,len(signal)-2):
		sec_diff.append(abs(signal[i+2]-signal[i]))

	fd_sum = sum(sec_diff)
	delta = float(fd_sum)/(len(signal)-2)
	return(delta)

def norm_second_differences(signal):
	"""The mean of the absolute values of the second differences of the normalized signal"""
	signal_normed = normalise_signal(signal)
	sec_diff = []
	for i in range(0,len(signal_normed)-2):
		sec_diff.append(abs(signal_normed[i+2]-signal_normed[i]))

	fd_sum = sum(sec_diff)
	delta = float(fd_sum)/(len(signal_normed)-2)
	return(delta)

def package_deal_signal(signal):
	"""Returns the package of normalised mean, stdev, normalized first and second differences"""
	normedSig = normalise_signal(signal)
	m = mean(normedSig)
	s = std_dev(normedSig)
	nfd = norm_first_differences(signal)
	nsd = norm_second_differences(signal)

	return m,s,nfd,nsd

def data_binarizer(ratings, threshold):
	"""binarizes the data below and above the threshold"""
	binarized = []
	for rating in ratings:
		if rating <= threshold:
			binarized.append(0)
		else:
			binarized.append(1)
	return binarized
