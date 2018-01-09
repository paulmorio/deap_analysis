# EEG preprocessing
import numpy as np
import scipy.signal

def lr_assymetry(eeg_dict):
	# asymmetry between left-right prefrontal areas, related to approach/withdrawal
	# Prefrontal Cortex, Emotion, and Approach/Withdrawal Motivation
	# Jeffrey M. Spielberg, Jennifer L. Stewart, Rebecca L. Levin, Gregory A. Miller, and Wendy Heller

	# Left Nodes = 
	# Right Nodes =

def eeg_bands(signal):
	# returns the alpha, theta, beta
	eeg = signal  
	pi = 3.14
	y = np.array(eeg)  # faster array
	L = len(eeg)       # signal length
	fs = 128.0     # frequency sampling 128Hz
	T = 1/fs           # sample time
	t = np.linspace(1,L,L)*T

	f = fs*np.linspace(0,L/10,L/10)/L  # single side frequency vector, real frequency up to fs/2
	Y = np.fft.fft(y)

	filtered = []
	b= [] # store filter coefficient
	cutoff = [0.5,4.0,8.0,12.0,30.0] # delta, alpha, theta, beta bands

	for band in xrange(0, len(cutoff)-1):
		wl = 2*cutoff[band]/fs*pi
		wh = 2*cutoff[band+1]/fs*pi
		M = 512      # Set number of weights as 128
		bn = np.zeros(M)

		for i in xrange(0,M):     # Generate bandpass weighting function
			n = i-  M/2       # Make symmetrical
			if n == 0:
				bn[i] = wh/pi - wl/pi;
			else:
				bn[i] = (np.sin(wh*n))/(pi*n) - (np.sin(wl*n))/(pi*n)   # Filter impulse response

		bn = bn*np.kaiser(M,5.2)  # apply Kaiser window, alpha= 5.2
		b.append(bn)

		[w,h]= scipy.signal.freqz(bn,1)
		filtered.append(np.convolve(bn, y)) # filter the signal by convolving the signal with filter coefficients

	alpha = filtered[1]
	theta = filtered[2]
	beta = filtered[3]

	return alpha, theta, beta

def eeg_w_beta(signal):
	# we calculate the relative logged power of the theta signal
	# which has been shown to be correlated to workload, attention, valence*
	# Mental Fatigue Measurement Using EEG Cheng et al.
	alpha, theta, beta = eeg_bands(signal)
	nomin = np.sum(beta)
	denom = np.sum(theta + alpha)
	eeg_w = np.log(nomin/denom)
	return eeg_w


