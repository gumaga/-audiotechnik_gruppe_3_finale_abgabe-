"""
script classifier_speech_features.py
This file may be run as a script or used as a class.
For information on this file see: README.md

Authors: Leon Hochberger, Daniel-José Alcala Padilla, Tobias Danneleit   
Date: March 10th, 2021
License: 3-clause BSD (see: https://github.com/gumaga/-audiotechnik_gruppe_3_finale_abgabe-)
"""

import soundfile as sf
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt 
import os
import compressor   # module containing compressor class
import filterbank   # module containing filterbank class
import pickle
from python_speech_features import mfcc
from python_speech_features import sigproc
from sklearn.svm import SVC
from pathlib import Path

class classifier_speech_features():

    def __init__(self):
        self.block_len_ms = 30
        self.overlap = 0.5

    def crest_factor(self, data):
        rms = np.sqrt(np.mean((data)**2))
        cf = np.abs(np.max(data))/rms
        return cf

    def load_data(self, label, pretrained_mode=False):
        # load files out of one folder
        directory = os.path.dirname(os.path.abspath(__file__)) # directory containing this file

        if pretrained_mode == True: 
            data_path = os.path.join(directory, label) # path of folder containing test-data
        else:    
            data_path = os.path.join(directory, "Dataset_small", label)   # path of subfolders containing train-data
        folder = Path(data_path).rglob('*.wav')
        files = [x for x in folder] # all files inside a folder with global path

        print('done loading')
        return files

    def feature_extraction(self, files, feature, return_testrun_values=False ):

        files_extracted = []

        for filename in files:
            signal, fs = sf.read(filename) # for signal-length and samplerate

            if return_testrun_values == True: # only for test signal: (in case of stereo signal)
                signal = signal[:,1] # only left channel
                signal_len = len(signal)
            else:
                None

            block_length_samples = int(np.floor((self.block_len_ms / 1000) * fs))
            forward = 1 - self.overlap
            overlap_samples = int(np.floor(self.overlap * block_length_samples))
            n_blocks = int(np.floor(len(signal)/overlap_samples)) #number of blocks for current wav-file

            signal_blocks = sigproc.framesig(signal, block_length_samples, forward * block_length_samples)

            if feature == 'mfcc':
                feat_block = np.zeros((int(n_blocks),13)) # preallocating: n_blocks x 13 because we will have 13 MFCC-Values
            else:
                feat_block = np.zeros((int(n_blocks),1)) # preallocating
            
            nfft = int(2**(np.ceil(np.log2(block_length_samples)))) # nfft = next greatest integer value to the power of 2 (Zeropadding)

            n = 0 # count blocks 
            for block in signal_blocks:
                if feature == 'mfcc':
                    feat_block[n] = mfcc(block,fs, winlen=self.block_len_ms/1000, winstep=0.001, nfft=nfft) # extract feature per block
                else:
                    feat_block[n] = self.crest_factor(block) # extract feature per block
                n += 1 # increment counter

            files_extracted.append(feat_block) # list of arrays: one array contains blockwise features of one audiofile

        print('Done Extracting')
        if return_testrun_values == True:
            return files_extracted, signal, fs, signal_len, block_length_samples, forward
        else:    
            return files_extracted


    def train_classifier(self, classifier_type='svm'): 
        '''
        Classifier: svm = support vector machines
                    gmm = gaussian mixture models
                    logReg = logistic regression
        Classes: Y = labels
            speech = class 1
            not speech = class 0
        '''

        # training data with labels
        if classifier_type == 'svm':
            classifier = SVC()
        elif classifier_type == 'gmm':
            classifier = GaussianMixture()
        elif classifier_type == 'logReg':
            classifier = LogisticRegression()
        else:
            None

        # concatenate all feature vectors x and labels y
        x_noise = np.vstack((noise_feat[0:-1]))
        x_speech = np.vstack((speech_feat[0:-1]))
        x_music = np.vstack((music_feat[0:-1]))
        X_train = np.array(np.vstack((x_noise, x_speech, x_music))) # training data

        y_noise = np.zeros((len(x_noise),1))
        y_speech = np.ones((len(x_speech),1))
        y_music = np.zeros((len(x_music),1))
        Y_train = np.array(np.vstack((y_noise, y_speech, y_music)))
        Y_train.shape = (len(Y_train))          # labels of training data

        classifier.fit(X_train, Y_train) # train classifier
        print("classifier trained / fitted")
        return classifier


if __name__ == "__main__":
    classifier_functions = classifier_speech_features()

    feature = 'mfcc' # alternatively: 'cf' for crest factor

    # loading training data and exracting features and training classifier
    noise_files = classifier_functions.load_data('Noise') 
    noise_feat = classifier_functions.feature_extraction(noise_files, feature)

    speech_files = classifier_functions.load_data('Speech') 
    speech_feat = classifier_functions.feature_extraction(speech_files, feature)

    music_files = classifier_functions.load_data('Music')
    music_feat = classifier_functions.feature_extraction(music_files, feature)

    classifier_svm = classifier_functions.train_classifier(classifier_type='svm')     

    # save classifier object to file into same directory as this python-file
    directory = os.path.dirname(os.path.abspath(__file__)) # directory containing this file
    data_path = os.path.join(directory, "classifier_file.obj")   # new file containing classifier
    with open(data_path, 'wb') as pfile:
        pickle.dump(classifier_svm, pfile) 
        print('saved file')
