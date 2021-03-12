"""
class for designing a 3-band filterbank 

Authors: Leon Hochberger, Daniel-José Alcala Padilla, Tobias Danneleit   
Date: March 10th, 2021
License: 3-clause BSD (see README in https://github.com/gumaga/-audiotechnik_gruppe_3_finale_abgabe-)
"""
import scipy.signal as signal
import numpy as np 

class filterbank():
    def __init__(self, fs, lower_cutoff = 500, upper_cutoff=2500):
        # create filterbank with default cutoff frequencies 500 and 2500 Hz 
        cutoffs = np.array([lower_cutoff, upper_cutoff])
        filter_order = 4
        self.b_1, self.a_1 = signal.butter(N=filter_order, Wn=lower_cutoff, fs=fs) #lowpass
        self.b_2, self.a_2 = signal.butter(N=filter_order, Wn=2*cutoffs/fs, btype='bandpass') #bandpass
        self.b_3, self.a_3 = signal.butter(N=filter_order, Wn=2*upper_cutoff/fs, btype='high') #highpass

    def filter(self, signal_in):
        # filter signal in three pieces and return time signal of filtered signal per band
        fltrd_sig_1 = signal.filtfilt(self.b_1, self.a_1, signal_in, axis=0)# lowpass-filtered
        fltrd_sig_2 = signal.filtfilt(self.b_2, self.a_2, signal_in, axis=0)# bandpass-filtered
        fltrd_sig_3 = signal.filtfilt(self.b_3, self.a_3, signal_in, axis=0)# highpass-filtered
        return fltrd_sig_1, fltrd_sig_2, fltrd_sig_3
