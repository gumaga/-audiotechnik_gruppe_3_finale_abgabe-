'''
class for designing a compressor. Implementation is based on [1].

Authors: Leon Hochberger, Daniel-José Alcala Padilla, Tobias Danneleit   
Date: March 10th, 2021
License: 3-clause BSD (seesee README in https://tgm-git.jade-hs.de/leon.hochberger/audiotechnik_gruppe_3_finale_abgabe.git)

Sources:
[1] Giannoulis, D., Massberg, M., und Reiss, J. D. (2012).                
   Digital Dynamic Range Compressor Design-A Tutorial and Analysis. 
   J. Audio Eng. Soc, 60(6):399–408.
'''
import numpy as np

class compressor():
    def __init__(self, fs, thresh=-20, ratio=2, t_attack=0.01, t_release=0.05, mu_auto=True, soft_knee=True):

        self.fs = fs
        self.thresh = thresh
        self.ratio = ratio
        self.mu_auto = mu_auto
        self.soft_knee = soft_knee

        self.alpha_attack = np.exp(-1/ (t_attack * fs))
        self.alpha_release = np.exp(-1/ (t_release * fs))

    def compress_mono(self, signal, compress_indices=None):

        # indices where to compress the signal. Default: Complete Signal is beeing compressed
        if compress_indices is None:
            compress_indices = np.ones(len(signal))

        # Presets to soft knee
        if self.soft_knee == True:
            k = 4             # parameter for soft knee
            k_over_thres = self.thresh + k + (1/self.ratio - 1) * (self.thresh + k - self.thresh + k)**2 /(4*k) # auxillary variable
        else:        # If soft knee is turned off (False)   
            k = 0
            k_over_thres = self.thresh # auxillary variable

        # auto makeup gain
        if self.mu_auto == True:
            mu_gain = self.thresh*(1/self.ratio - 1)       
        else:
            mu_gain = 0

        # keep the shape of input signal
        shape = signal.shape
        
        # square of input signal out of which the leading signal 'steuer_out' will be generated
        square_in = signal**2
        
        # preallocating  
        steuer_out = np.zeros(shape)
        steuer = np.zeros(shape)
        gain_dB = np.zeros(shape)
            
        
        for ii in np.arange(0,len(steuer_out)):
            
            # samples with values of zero, will be set to a very small value close to zero
            if square_in[ii] == 0: 
                square_in[ii] = 10**(-30)
            else:
                None
            
            # if signal should not be compressed, set squared input to a very low value so there 
            # wont be any compression but also no crackling sound
            if compress_indices[ii] == 0:
                square_in[ii] = 10**(-30) 


            # generate leading signal
            if ii == 0:    
                steuer[ii] =  square_in[0] # 1st value of leading signal
            else:
                # apply attack and release time and prevent oscillation ######### so richtig?
                if square_in[ii] > steuer[ii-1] * self.alpha_attack:    
                    alpha = self.alpha_attack
                elif square_in[ii] < steuer[ii-1] * self.alpha_release:
                    alpha = self.alpha_release
                    
                steuer[ii] = (alpha * (steuer[ii-1] - square_in[ii])) + square_in[ii]   

            # leading signal in dB
            steuer_out[ii] = 10*np.log10(steuer[ii])

            # determine gain according to level of leadig signal
            if steuer_out[ii] <= self.thresh - k: # if input is in linear section
                gain_dB[ii] = 0

            elif steuer_out[ii] > self.thresh - k and steuer_out[ii] < self.thresh + k:# if input is in soft knee section
            
                gain_dB[ii] = (1/self.ratio - 1) * (steuer_out[ii] - self.thresh + k)**2 /(4*k) 
                
            else: # if input is in section of constant ratio
                gain_dB[ii] = (steuer_out[ii] - (self.thresh + k))/ self.ratio + k_over_thres - steuer_out[ii]

        gain_dB += mu_gain
        signal_out = signal * 10**(gain_dB/20)
    
        # Peak-Clipping
        signal_out_clip = self.peak_clipper(signal_out)

        return signal_out_clip

    def peak_clipper(self, signal):
        clipped = np.zeros((len(signal),1))
        for jj in np.arange(0,len(signal)):
            if signal[jj] >= 1: # values of 1 or higher will be output at value 1
                clipped[jj] = 1
            elif signal[jj] <= -1: # values of -1 or lower will be output at value -1
                clipped[jj] = -1
            else: # values between -1 and 1 will be set according to a 3rd degree polinomial 
                clipped[jj] = -0.5*(signal[jj]**3) + 1.5*signal[jj]
        return clipped
