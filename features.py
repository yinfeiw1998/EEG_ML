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

from scipy.signal import butter, lfilter
import pywt

from scipy import signal


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
        
    def write_in(self, user_id, file_name):
        f_object = open(file_name, "a")
#         f_object.write("id | rms | ApprEnt | LZComp | HP_activity | HP_mobility | HP_complexity | AvgFrequency | mpf | sef\n")
        
        f_object.write(str(user_id) + " | ")
        for i in range(len(self.channel)):
            f_object.write(str(self.rms[i]) + " ")
        f_object.write("| ")
        for i in range(len(self.channel)):
            f_object.write(str(self.ApprEnt[i]) + " ")
        f_object.write("| ")
        for i in range(len(self.channel)):
            f_object.write(str(self.LZ_comp[i]) + " ")
        f_object.write("| ")
        for i in range(len(self.channel)):
            f_object.write(str(self.HP_activity[i]) + " ")
        f_object.write("| ")
        for i in range(len(self.channel)):
            f_object.write(str(self.HP_mobility[i]) + " ")
        f_object.write("| ")
        for i in range(len(self.channel)):
            f_object.write(str(self.HP_complexity[i]) + " ")
        f_object.write("| ")
        for i in range(len(self.channel)):
            f_object.write(str(self.Afrequency[i]) + " ")
        f_object.write("| ")
        for i in range(len(self.channel)):
            f_object.write(str(self.TotalEnergy[i]) + " ")
        f_object.write("| ")
        for i in range(len(self.channel)):
            f_object.write(str(self.DeltaPower[i]) + " ")
        f_object.write("| ")
        for i in range(len(self.channel)):
            f_object.write(str(self.ThetaPower[i]) + " ")
        f_object.write("| ")
        for i in range(len(self.channel)):
            f_object.write(str(self.AlphaPower[i]) + " ")
        f_object.write("| ")
        for i in range(len(self.channel)):
            f_object.write(str(self.BetaPower[i]) + " ")
        f_object.write("| ")
        for i in range(len(self.channel)):
            f_object.write(str(self.DeltaRatio[i]) + " ")
        f_object.write("| ")
        for i in range(len(self.channel)):
            f_object.write(str(self.BARatio[i]) + " ")
        f_object.write("| ")
        for i in range(len(self.channel)):
            f_object.write(str(self.BetaRatio[i]) + " ")
        f_object.write("| ")
        for i in range(len(self.channel)):
            f_object.write(str(self.mpf[i]) + " ")
        f_object.write("| ")
        for i in range(len(self.channel)):
            f_object.write(str(self.sef[i]) + " ")
        f_object.write("| \n")
        f_object.close()
        
        


    def extract_features(self, user_id, start, end, file_name):
        """
        Beginning Feature Extraction
        Time Domain Features
        """
        
#         f_object = open(file_name, "a")
#         f_object.write("id | rms | ApprEnt | LZComp | HP_activity | HP_mobility | mpf | sef\n")
        
#         f_object.write(str(user_id) + " | ")
            
        indexes = [self.raw.ch_names.index(ch) for ch in self.channel]
        
        self.raw_control=(self.raw_eeg[indexes,start:end]) # choose ['F3-A2', 'F4-A1', 'C3-A2', 'C4-A1', 'O1-A2','O2-A1'] these channels from raw data
        self.sfreq = 200
        
        
        """
        rms
        """
        self.rms = arr.array('d')
        for i in range(len(self.channel)):
            if (np.sqrt(np.mean(self.raw_control[i]**2))<0):
                print("got you")
            self.rms.append(np.sqrt(np.mean(self.raw_control[i]**2)))
#             f_object.write(str(self.rms[i]) + " ")
#         f_object.write("| ")
            
#         print('raw_control', self.raw_control)
        
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
        for i in range(len(self.channel)):
            self.ApprEnt.append(ApEn(self.raw_control[i], 2, 3))
#             f_object.write(str(self.ApprEnt[i]) + " ")
#         f_object.write("| ")
            
            
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
        for i in range(len(self.channel)):
            self.LZ_comp.append(LZComplexity(self.raw_control[i], 3.0E-5))
#             f_object.write(str(self.LZ_comp[i]) + " ")
#         f_object.write("| ")
        
        """
        Hjorth Parameter
        """
        """
        Activity(Variance)
        """
        self.HP_activity = arr.array('d')
        for i in range(len(self.channel)):
            mean = np.mean(self.raw_control[i])
            self.HP_activity.append(np.mean((self.raw_control[i]-mean)**2))
#             f_object.write(str(self.HP_activity[i]) + " ")
#         f_object.write('| ')
        
        """
        Mobility
        """
        def mobility(lst):
            l = lst.shape[0]
            DR = lst[1:l] - lst[0:l-1]
            return np.sqrt(self.HP_activity[i]) / np.sqrt(np.mean((DR-mean)**2))
            
        self.HP_mobility = arr.array('d')
        for i in range(len(self.channel)):
#             l = self.raw_control[i].shape[0]
#             DR = self.raw_control[i, 1:l] - self.raw_control[i, 0:l-1]
#             dmean = np.mean(DR)
#             self.HP_mobility.append(np.sqrt(self.HP_activity[i]) / np.sqrt(np.mean((DR-mean)**2)))
            self.HP_mobility.append(mobility(self.raw_control[i]))
#             f_object.write(str(self.HP_mobility[i]) + " ")
#         f_object.write('| ')
        
        """
        Complexity
        """
        self.HP_complexity = arr.array('d')
        for i in range(len(self.channel)):
            l = self.raw_control[i].shape[0]
            DR = self.raw_control[i, 1:l] - self.raw_control[i, 0:l-1]
            self.HP_complexity.append(mobility(DR)/mobility(self.raw_control[i]))
#             f_object.write(str(self.HP_complexity[i]) + " ")
#         f_object.write('| ')
        

        """
        Average Frequency
        """
        self.Afrequency = arr.array('d')
        for i in range(len(self.channel)):
            s = 0
            l = self.raw_control[i].shape[0]
            for j in range(l-1):
                if self.raw_control[i][j]>=0 and self.raw_control[i][j+1]<0:
                    s += 1
            self.Afrequency.append(s/(end-start))
#             f_object.write(str(self.Afrequency[i]) + " ")
#         f_object.write('| ')
        
        """
        Frequency Domain Features
        """
        """
        Convert to frequency domain using FFT
        """
            
        # used exmple from https://alphabold.com/fourier-transform-in-python-vibration-analysis/
        N = 10 * self.sfreq  #5 seconds * 200 samples/sec
#         frequency = np.linspace(0.0, int(self.sfreq/2), int(N/2))
        freq_data = [[0] * np.int(N/2)] * 6
        freqs = []
        for i in range(0, 6):
            y = fft(self.raw_control[i])
            freq_data[i] = (2/N * np.abs(y[0:np.int(N/2)]))
            
#         print('freqs', freqs)
        
#         print(freq_data)
            
        #plt.plot(frequency, y)
        #plt.title('Frequency domain Signal')
        #plt.xlabel('Frequency in Hz')
        #plt.ylabel('Amplitude')
        #plt.show()
        #length1 = len(y1)
        #length2 = len(frequency)
        #print("Length 1 " + str(length1))
        #print("Length 2 " + str(length2))
        
#         def butter_bandpass(lowcut, highcut, fs, order):
#             nyq = 0.5 * fs
#             low = lowcut / nyq
#             high = highcut / nyq
#             b, a = butter(order, [low, high], btype='band', analog=False)
#             return b, a
        
#         def butter_bandpass_filter(data, lowcut, highcut, fs, order):
#             b, a = butter_bandpass(lowcut, highcut, fs, order)
#             y = lfilter(b, a, data)
#             return y
        
#         def preprocess(extract, fs, total):
#             s1 = (len(extract), total)
#             fil_delta = np.zeros(s1)
#             fil_theta = np.zeros(s1)
#             fil_alpha = np.zeros(s1)
#             fil_beta = np.zeros(s1)
#             fil_all = np.zeros(s1)
# #             fil_gama = np.zeros(s1)
# #             fil_all = np.zeros(s1)
#             for i in range(len(extract)):
#                 fil_delta[i] = butter_bandpass_filter(extract[i], 0.5, 4, fs, 6)
#                 fil_theta[i] = butter_bandpass_filter(extract[i], 4, 7.5, fs, 6)
#                 fil_alpha[i] = butter_bandpass_filter(extract[i], 8, 16, fs, 6)
#                 fil_beta[i] = butter_bandpass_filter(extract[i], 16.5, 25, fs, 6)
#                 fil_all[i] = butter_bandpass_filter(extract[i], 0.5, 25, fs, 6)
#             return fil_delta, fil_theta, fil_alpha,fil_beta, fil_all
        
#         def feature_Extraction(extract, fil_delta, fil_theta, fil_alpha, fil_beta, fil_all):
#             fft_delta = [0]*len(extract)
#             fft_theta = [0]*len(extract)
#             fft_alpha = [0]*len(extract)
#             fft_beta = [0]*len(extract)
#             fft_all = [0]*len(extract)
#             for i in range(len(extract)):
#                 fft_delta[i] = (abs(np.fft.fft(fil_delta[i])))**2
#                 fft_theta[i] = (abs(np.fft.fft(fil_theta[i])))**2
#                 fft_alpha[i] = (abs(np.fft.fft(fil_alpha[i])))**2
#                 fft_beta[i] = (abs(np.fft.fft(fil_beta[i])))**2
#                 fft_all[i] = (abs(np.fft.fft(fil_all[i])))**2
#             mean_delta = np.mean(fft_delta, axis=1)
#             mean_theta = np.mean(fft_theta, axis=1)
#             mean_alpha = np.mean(fft_alpha, axis=1)
#             mean_beta = np.mean(fft_beta, axis=1)
#             mean_all = np.mean(fft_all, axis=1)
            
#             return mean_delta, mean_theta, mean_alpha, mean_beta, mean_all
        
        #https://raphaelvallat.com/bandpower.html refers to this website
        def bandpower(data, sf, band, relative=False, totalpower = False):
            from scipy.signal import welch
            from scipy.integrate import simps
            band = np.asarray(band)
            low, high = band
            
            nperseg = sf
            
            freqs, psd = welch(data, sf, nperseg=nperseg)
            freq_res = freqs[1] - freqs[0]
            idx_band = np.logical_and(freqs>=low, freqs<=high)
            bp = simps(psd[idx_band], dx=freq_res)
            
            if relative:
                bp /= simps(psd, dx=freq_res)
            if totalpower:
                return simps(psd, dx=freq_res)
            
            return bp
        
        """
        Total Energy/Power
        Band Power
        Delta Power (0.5-4)
        Theta Power (4-8)
        Alpha Power (8-12)
        Beta Power (12-30)
        Delta Ratio
        BA Ratio
        Beta Ratio
        """
        self.DeltaPower = []
        self.ThetaPower= []
        self.AlphaPower= []
        self.BetaPower= []
        self.TotalEnergy= []
        
        self.DeltaRatio = []
        self.BARatio = []
        self.BetaRatio = []
        
#         fil_delta, fil_theta, fil_alpha,fil_beta, fil_all = preprocess(self.raw_control, self.sfreq, (end-start))
#         mean_delta, mean_theta, mean_alpha, mean_beta, mean_all = feature_Extraction(self.raw_control, fil_delta, fil_theta, fil_alpha, fil_beta, fil_all)
#         print('start FFT')
        
        for i in range(len(self.channel)):
            self.TotalEnergy.append(bandpower(self.raw_control[i], self.sfreq, (0, 100), totalpower=True))
            self.DeltaPower.append(bandpower(self.raw_control[i], self.sfreq, (0.5, 4), relative=True))
            self.ThetaPower.append(bandpower(self.raw_control[i], self.sfreq, (4, 7.5), relative=True))
            self.AlphaPower.append(bandpower(self.raw_control[i], self.sfreq, (8, 16), relative=True))
            self.BetaPower.append(bandpower(self.raw_control[i], self.sfreq, (16, 25), relative=True))
            
            self.DeltaRatio.append(bandpower(self.raw_control[i], self.sfreq, (8, 25)) / self.DeltaPower[i])
            self.BARatio.append(self.BetaPower[i]/self.AlphaPower[i])
            self.BetaRatio.append(math.log10(bandpower(self.raw_control[i], self.sfreq, (13, 30), relative=True)))
            
#         print(bandpower(self.raw_control[0], self.sfreq, (0.5, 4)))
#         for i in range(len(self.channel)):
#             TotalPower = (abs(np.fft.fft(fil_theta[i])))**2
            
#             self.DeltaPower.append(fil_delta[i]/fil_all[i])
#             self.ThetaPower.append(fil_theta[i]/fil_all[i])
#             self.AlphaPower.append(fil_Alpha[i]/fil_all[i])
#             self.BetaPower.append(fil_Beta[i]/fil_all[i])
            
        def getfreq(extract, start_freq, end_freq, fs, total):
            sl = (len(extract), total)
            fil = np.zeros(s1)
            for i in range(len(extract)):
                fil[i] = butter_bandpass_filter(extract[i], start_freq, end_freq, fs, 6)
            return fil
                
        
#         """
#         Delta Ratio
#         """
#         self.DeltaRatio = []
#         for i in range(len(self.channel)):
#             self.DeltaRatio.append((np.sum([i**2 for i in idx_Alpha])+np.sum([i**2 for i in idx_Beta]))/np.sum(idx_Delta))
            
#         """
#         BA Ratio
#         """
#         self.BARatio = []
#         for i in range(len(self.channel)):
#             self.BARatio.append(np.sum([i**2 for i in idx_Beta])/np.sum(idx_Alpha))
            
#         """
#         Beta Ratio (deleted --> some data does not have Beta data)
#         """
#         self.BetaRatio = []
#         for i in range(len(self.channel)):
#             print(np.sum(idx_Beta)/self.TotalEnergy[i])
#             self.BetaRatio.append(math.log10(np.sum(idx_Beta)/self.TotalEnergy[i]))
        
        
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
#             f_object.write(str(self.mpf[i]) + " ")
#         f_object.write("| ")

            
        for i in range(0, 6):
            self.sef[i] = (bin_search(freq_data[i], 0.95))
#             f_object.write(str(self.sef[i]) + " ")
#         f_object.write("\n")
        
#         f_object.close()

        self.write_in(user_id, file_name)
        
        
        
        """
        According to the article, the top 5 features are entropy, LZ complexity, mpf, sef, and rms
        """
        