# peripheral features
import numpy as np
import scipy.signal

def power_spectrums(signal):
	# calculates the concatenated vector of the power spectra
	# for the alpha, theta, beta bands
	# via the welch method for estimating spectral density
	c = scipy.signal.welch(signal, fs=128, scaling = 'spectrum')
	interestingPower = c[1]
	return interestingPower