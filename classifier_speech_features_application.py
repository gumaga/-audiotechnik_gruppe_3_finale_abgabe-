"""
Script classifier_speech_features_application.py
For more information on this file see: README.md

Authors: Leon Hochberger, Daniel-José Alcala Padilla, Tobias Danneleit   
Date: March 10th, 2021
License: 3-clause BSD (see README in https://tgm-git.jade-hs.de/leon.hochberger/audiotechnik_gruppe_3_finale_abgabe.git)
"""

import soundfile as sf
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt 
import os
import pickle
import compressor   # module containing compressor class
import filterbank   # module containing filterbank class
import classifier_speech_features as csf
from pathlib import Path

# object for using functions to load data and extract features
classifier_functions = csf.classifier_speech_features()

# load pretrained classifier
directory = os.path.dirname(os.path.abspath(__file__)) # directory containing this file
data_path = os.path.join(directory, "classifier_file.obj")
data_path_shelve = os.path.join(directory, "classifier_file_shelve")
with open(data_path, 'rb') as pfile:
    classifier = pickle.load(pfile)

# load audio file which is going to be classified and compressed
test_files = classifier_functions.load_data('test_files', pretrained_mode=True)
desired_feature = 'mfcc' 
feature_variables = classifier_functions.feature_extraction(test_files, desired_feature, return_testrun_values=True)
# unpack touple for better readability
features, signal, fs, signal_length, block_length, forward_length = feature_variables
features = features[0]  # get first array out of list, to avoid errors. 
                        # No need for multiple arrays because we only want to read one file 

# classify loaded audio file 
Y_predict = classifier.predict(features)

# to determine where the compressor should work a leading signal is being
# created. If speech was predicted in 3 out of 4 overlapping blocks, 2
# succesive blocks (i.e. 60 ms) are marked as speech.
blocks_without_overlap = int(np.ceil(signal_length/block_length)) # nr of blocks without overlap
lead = []
ones_vec = np.ones((block_length*2, 1))
zeros_vec = np.zeros((block_length*2, 1))

zz = 0 # counter
for value in np.arange(int(np.ceil(blocks_without_overlap/2))):
    if np.sum(Y_predict[zz:zz+4]) >= 3: 
        lead = np.append(lead, ones_vec)
    else:                    
        lead = np.append(lead, zeros_vec)
    zz += 4
lead = np.array(lead)    

# get lead-vector that has same length as signal, because we are calculating
# a little bit more than that
lead = lead[0:signal_length]

# divide signal into three filter-bands
lower_border_freq = 250
upper_border_freq = 4000
filt = filterbank.filterbank(fs, lower_border_freq, upper_border_freq)
low_sig, mid_sig, high_sig = filt.filter(signal)

# compress signal in filter-bands depending on where speech was detected
comp = compressor.compressor(fs=fs, ratio=2, mu_auto=True)
gain = 1/4 # -12 dB
compressed_low = comp.compress_mono(low_sig, lead)
compressed_mid = comp.compress_mono(mid_sig, lead)
compressed_high = comp.compress_mono(high_sig, lead)
signal_out = gain * compressed_low + compressed_mid + gain * compressed_high

# write signal into file so that you can (hopefully) hear a difference
sf.write('test_signal_compressed.wav', data=signal_out, samplerate=fs)

########
'''
Ideen: subplot mit time-signal oben, (prediction) in der Mitte oder als Linie im oberen Plot und unten was wir daraus gemacht haben-> lead_signal
'''