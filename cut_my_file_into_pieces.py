"""
Script for cutting wav.-files into 3 second long pieces.
Each piece is saved as its own wav.-file.

Original files which are to be cut must be stored inside a specific structure of folders.
It uses a main folder named 'Dataset_new' with the three subfolders 'Speech', 'Noise' and 'Music'.
The cut pieces are stored in folders of the same structure. 
Here the main folder is named 'Dataset_new_cut', while the three subfolders remain the same.
In order to use this script the main folders must be created in the same directory as this script.

Authors: Leon Hochberger, Daniel-José Alcala Padilla, Tobias Danneleit   
Date: March 10th, 2021
License: 3-clause BSD (see README in https://tgm-git.jade-hs.de/leon.hochberger/audiotechnik_gruppe_3_finale_abgabe.git)
"""
import os
import soundfile as sf
from python_speech_features import sigproc
from pathlib import Path


file_len_s = 3 # length in seconds to which a file is to be cut to 
directory = os.path.dirname(os.path.abspath(__file__)) # directory containing this file

# run through the three subfolders inside Dataset_new folder
for label in {'Music', 'Noise', 'Speech'}:

    data_path = os.path.join(directory, "Dataset_new", label) # path of subfolders containing data
    folder = Path(data_path).rglob('*.wav')       
    files = [x for x in folder] # all files inside a folder with global path

    ii = 0
    for filename in files:
        signal, fs = sf.read(filename) #loading original files 
        cut_signals = sigproc.framesig(signal, file_len_s*fs, file_len_s*fs) # cut in blocks

        kk = 0
        for sig in cut_signals:
            file_name = label + str(ii) + "cut" + str(kk) + ".wav"
            cut_file_path = os.path.join(directory, "Dataset_new_cut", label, file_name) # path where file will be saved to
            sf.write(cut_file_path, sig, fs)
            kk +=1
        ii += 1

