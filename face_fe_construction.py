# facial embedding fe construction

# Affect recognition via EEG
import numpy as np
import signalpreprocess as sp
import peripheral_features as pf
import cPickle
import sys
import glob
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('imgs', type=str, nargs='+', help="Input image representations per participant.")
args = parser.parse_args()

args.imgs[0]
embeddingsDict = cPickle.load(open(args.imgs[0], 'rb'))

