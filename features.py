import pandas as pd
import numpy as np
import pyedflib
import mne
import array as arr
import statistics
import math
import matplotlib.pyplot as plt
from scipy import pi
from scipy.fftpack import fft
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs)


class features:
    
    def __init__(self, file):
        self.raw = mne.io.read_raw_edf("../Human_data/" + file + ".edf", preload=True)
        self.channel=['F3-A2', 'F4-A1', 'C3-A2', 'C4-A1', 'O1-A2','O2-A1']
        

        #raw.drop_channels(['EOG-L','EOG-R','Chin','R Leg','L Leg','ECG','C-FLOW','Chest','Abdomen','Snore','SpO2','Airflow','P-Flo','R-R','C-Press','EtCO2']) 
        
        
        """
        ICA preprocessing
        """
        
        
        #ICA does not work with low frequencies, so a high pass filter is applied
#         raw_tmp = raw.copy()
#         raw_tmp.filter(1, None)
        
#         #instantiate an ICA object. infomax is reccommended according to https://cbrnr.github.io/2018/01/29/removing-eog-ica/.
#         #random state is assigned to one simply for reproducable results as otherwise it is randomized.
#         #Unclear if it should be anything else at the moment.
#         ica = mne.preprocessing.ICA(method="infomax",
#                                     fit_params={"extended": True},
#                                     random_state=1)
        
#         ica.fit(raw_tmp)
        
        
        
#         """
#         Artifact detection
#         """
        
        
#         ica.exclude = []
        
#         eog_l_indices, eog_l_scores = ica.find_bads_eog(raw, 'EOG-L')
#         eog_r_indices, eog_r_scores = ica.find_bads_eog(raw, 'EOG-R')
#         ecg_indices, ecg_scores = ica.find_bads_ecg(raw, 'ECG')
#         ica.exclude = eog_l_indices + eog_r_indices + ecg_indices
        
        
        """
        Artifact Repair
        """
        
        #remove ica.exclude from the original raw
        reconst_raw = self.raw.copy()
        #ica.apply(reconst_raw) #comment this to decrease runtime

        
        self.raw_eeg = reconst_raw[:, :][0]

        


    def extract_features(self, user_id, start, end, file_name):
        """
        Beginning Feature Extraction
        Time Domain Features
        """
        
        f_object = open(file_name, "a")
        f_object.write("id | rms | ApprEnt | LZComp | mpf | sef\n")
        
        f_object.write(str(user_id) + " | ")
            
        indexes = [self.raw.ch_names.index(ch) for ch in self.channel]
        
        self.raw_control=(self.raw_eeg[indexes,start:end]) # choose ['F3-A2', 'F4-A1', 'C3-A2', 'C4-A1', 'O1-A2','O2-A1'] these channels from raw data
        self.sfreq = 200
        
        
        """
        rms
        """
        self.rms = arr.array('d')
        for i in range(0, 6):
            self.rms.append(np.sqrt(np.mean(self.raw_control[i]**2)))
            f_object.write(str(self.rms[i]) + " ")
        f_object.write("| ")
            
        
        """
        Approximate Entropy
        """
        def ApEn(U, m, r) -> float:
            """Approximate_entropy."""
        
            def _maxdist(x_i, x_j):
                return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
        
            def _phi(m):
                x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
                C = [
                    len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0)
                    for x_i in x
                    ]
                return (N - m + 1.0) ** (-1) * sum(np.log(C))
        
            N = len(U)
        
            return abs(_phi(m + 1) - _phi(m))
        
        
        self.ApprEnt = arr.array("d")
        #print("Approximate Entropy\n")
        for i in range(0,6):
            self.ApprEnt.append(ApEn(self.raw_control[i], 2, 3))
            f_object.write(str(self.ApprEnt[i]) + " ")
        f_object.write("| ")
            
            
        """
        LZ complexity
        """
        def LZComplexity(arr, thres) -> float:
            n = len(arr)-1
            i = 0
            complexity = 1
            prefix_length = 1
            component_length = 1
            max_component = component_length
            while prefix_length + component_length <= n:
               if (abs(arr[i + component_length]) > thres and abs(arr[prefix_length + component_length]) > thres or 
                   abs(arr[i + component_length]) <= thres and abs(arr[prefix_length + component_length]) <= thres):
                  component_length = component_length + 1
               else:
                  max_component = max(component_length , max_component)
                  i = i + 1
                  if i == prefix_length :  # all the pointers have been treated
                     complexity = complexity + 1
                     prefix_length = prefix_length + max_component
                     component_length = 1
                     i = 0
                     max_component = component_length
                     
                  component_length = 1
                      
                  
            if component_length != 1:
                complexity = complexity +1
                
            return complexity
        
        #print("LZ complexity\n")
        self.LZ_comp = arr.array('d')
        for i in range(0,6):
            self.LZ_comp.append(LZComplexity(self.raw_control[i], 3.0E-5))
            f_object.write(str(self.LZ_comp[i]) + " ")
        f_object.write("| ")
        
        
        """
        Frequency Domain Features
        """
        """
        Convert to frequency domain using FFT
        """
        # used exmple from https://alphabold.com/fourier-transform-in-python-vibration-analysis/
        N = 10 * self.sfreq  #5 seconds * 200 samples/sec
        frequency = np.linspace (0.0, int(self.sfreq/2), int(N/2))
        freq_data = [[0] * np.int(N/2)] * 6
        for i in range(0, 6):
            y = fft(self.raw_control[i])
            freq_data[i] = (2/N * np.abs(y[0:np.int(N/2)]))
            
        #plt.plot(frequency, y)
        #plt.title('Frequency domain Signal')
        #plt.xlabel('Frequency in Hz')
        #plt.ylabel('Amplitude')
        #plt.show()
        #length1 = len(y1)
        #length2 = len(frequency)
        #print("Length 1 " + str(length1))
        #print("Length 2 " + str(length2))
        
        
        
        """
        Median Power Frequency (MPF)
        and Spectral Edge Frequency (SEF)
        (Directions in paper were unclear - page 86)
        """
        self.mpf = [0]*6
        self.sef = [0]*6
        
        #power_array = [[0]*1000]*6 #using this array to keep track of total power at various frequencies

        for i in range (0, 6):
            for j in range(0, 1000):
                if(j == 0):
                    freq_data[i][0] = freq_data[i][0]**2
                else:
                    freq_data[i][j] = freq_data[i][j]**2 + freq_data[i][j-1]
            #print(str(i) + "|" + str(j) + ":" + str(freq_data[i][j]))
    
        def bin_search(arr, x) -> int:
            l = 0
            r = 999
            answer = 0
            while (l <= r):
                m = int((l + r)/2)
                ratio = arr[m]/arr[999]
                #print("m: " + str(m))
                #print("ratio: " + str(ratio))
                if(ratio < x):
                    answer = m
                    l = m + 1
                elif(ratio == x):
                    answer = m
                    l = r = m
                else:
                    r = m - 1
            return answer
            
        for i in range(0, 6):
            self.mpf[i] = (bin_search(freq_data[i], 0.5))
            f_object.write(str(self.mpf[i]) + " ")
        f_object.write("| ")

            
        for i in range(0, 6):
            self.sef[i] = (bin_search(freq_data[i], 0.95))
            f_object.write(str(self.sef[i]) + " ")
        f_object.write("\n")
        
        f_object.close()
        
        
        
        """
        According to the article, the top 5 features are entropy, LZ complexity, mpf, sef, and rms
        """
        