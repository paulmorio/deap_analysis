# peripheral features
import numpy as np
import scipy.signal
import signalpreprocess as sp

def power_spectrums(signal):
	# calculates the concatenated vector of the power spectra
	# for the alpha, theta, beta bands
	# via the welch method for estimating spectral density
	dk, baseline = scipy.signal.welch(signal[:384], fs=128, scaling = 'spectrum')
	dk, trial_freq = scipy.signal.welch(signal[5760:], fs=128, scaling = 'spectrum')
	c = trial_freq-baseline
	power = np.log(np.square(np.mean(c)))
	return power