# peripheral features
import numpy as np
import scipy.signal
import signalpreprocess as sp

def power_spectrums(signal):
	# calculates the concatenated vector of the power spectra
	# for the alpha, theta, beta bands
	# via the welch method for estimating spectral density
	c = scipy.signal.welch(signal, fs=128, scaling = 'spectrum')
	interestingPower = c[1]
	return interestingPower

def baselined_last_30_seconds(signal):
	pass