# here get all classes and function to pass to other scripts

import os
import math
from scipy import signal
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
import librosa, librosa.display
from matplotlib import pyplot as plt
from scipy.optimize import least_squares, curve_fit
import signal as sig
import glob
import itertools
from datetime import datetime
import sys
import pandas as pd
import operator
#import statisticz as st
import scipy
import shutil

class my_signals: #implements methods that all need
    def __init__(self):
        pass

   # def __mul__(self,other):
   #     return signal.convolve(self.data, other.data)
    
    

    def get_fft(self):
        self.ff_t=np.fft.fft(self.data,n=2**18)
        self.frequencies=np.fft.fftfreq(2**18,1/self.sampling_rate)
        return self.ff_t, self.frequencies
    
    def get_spectrum(self,spec_type='mel'):
        N, H = 4096, 512 #window and hop length
        if spec_type == 'mel':
            return librosa.feature.melspectrogram(
                self.data, self.sampling_rate, S=None,
                n_fft=N, hop_length=H, win_length=N
                ,window='hann',center=True,pad_mode='reflect',power=2.0)
        elif spec_type == 'cqt':
            return librosa.feature.chroma_cqt(self.data, self.sampling_rate, C=None,
                hop_length=H, fmin=None, norm=inf, threshold=0.0,
                tuning=None, n_chroma=12, n_octaves=7, window=N,
                bins_per_octave=None, cqt_mode='full')
        elif spec_type == 'stft':
            return librosa.stft(self.data, n_fft=N, hop_length=H,
                win_length=N, window='hanning')



    def get_max_freqs(self,ff_t=None):
        if ff_t is None:
            ff_t = self.ff_t
        freqs = self.frequencies
        max_peak = 0
        peaks, _  =scipy.signal.find_peaks(np.abs(ff_t),distance=100000)
    #    print(freqs[peaks], ff_t[peaks])
        try:
            temp = range(peaks[0]-1,peaks[0]+2)
            max_peak = average_peak(([np.abs(x) for x in ff_t[temp]]),list(freqs[temp]))
        except:
            print('exception in helper-note-instance get max_freqs occured')
     #   print(max_peak)
        return max_peak

class note_instance(my_signals):

    def __init__(self, name=None, midi_note=None
                ,fundamental=None, data=None, list_of_partials=None
                ,sampling_rate=44100, tablature_values = None, ff_t = None):

        self.sampling_rate = sampling_rate
        if name is None: #name is the track name as saved from crop?
            raise Exception('no name given for the note instance')
            os.exit(0)
        else:
            self.name = name
        if data is None: #data is the matrix got from librosa load
            self.data = self.load_track()
            self.get_fft()
        else:
            self.data = data
            self.get_fft()

        if midi_note is None:
            pass
        else:
            self.midi_note = midi_note
            self.fundamental = self.midi_to_fund()
        if fundamental is None:
            pass
        else:
            self.fundamental = fundamental
            self.midi_note = self.fund_to_midi()
        if list_of_partials is None:
            self.list_of_partials = []
        else:
            self.list_of_partials = list_of_partials
        if None in (self.fundamental,self.midi_note):
            raise Exception('not proper fundamental, midi note')
            exit(1)
        self.fundamental_measured = self.measure_fund()
        self.differences = []
        self.beta_list = []
        self.candidates = []
        self.tablature_values = tablature_values

    def measure_fund(self):
        zeroed = self.zero_out(self.fundamental,10)
        return self.get_max_freqs(zeroed)

    def load_track(self):
        data, fs = librosa.load(self.name,self.sampling_rate)
        return data
    
    def append_partial(self, partial):
        self.list_of_partials.append(partial)

    def midi_to_fund(self):
        return 440*2**((self.midi_note-69)/12)

    def fund_to_midi(self):
        return round(12*math.log(self.fundamental/440,2)+69)

    def calculate_cand_fund(self):
        ff_t, freqs = self.get_fft()
        f_bin = self.sampling_rate/ff_t.size
        ff_t = np.abs(ff_t)
        [norm_fft] = normalize([ff_t], norm = 'max', axis = 1)
        z_n_fft = list(filter(lambda x: x>0.01,norm_fft)) #zero out less than 1% samples
        peaks, _  =scipy.signal.find_peaks(z_n_fft) #get indexes of local maxima
        peaks = freqs[peaks]
        for x in peaks:
            if 80<x<600: #check if a valid freq
                temp = round(12*math.log(x/440,2)+69)
                if abs(temp-round(temp))<0.2:
                    self.candidates.append(x)

    def determine_fund(self):
        fscore_dict = {}
        for t in self.candidates:
            self.fundamental_measured = t
            self.differences = []
            self.list_of_partials = []
            self.compute_partials(5,40)
            fscore_dict[t] = sum(list(map(abs,self.differences)))
        print(fscore_dict)
        print(min(fscore_dict.items(), key=operator.itemgetter(1))[0])


    def zero_out(self,frequency,diviate):
        sz = self.ff_t.size
        x = np.zeros(sz,dtype=np.complex64)
        temp = (self.ff_t)
        dom_freq_bin = int(round(frequency*sz/44100))
        diviate = int(diviate)
        for i in range(dom_freq_bin-diviate,dom_freq_bin+diviate):
            x[i] = temp[i]**2
        return x



    def compute_partials(self, no_of_partials,diviate=65,mode='flat', cand_fund=None, beta=None, d=None, r=None): # flat compute
        '''compute partials given a fundamental frequency based on the mode needed.'''
        if mode =='flat': # compute beta without using beta at any stage. usually try for low order of partial
            diviate = round(diviate/(self.sampling_rate/self.ff_t.size))
            for i in range(2,no_of_partials):
                filtered = self.zero_out(i*self.fundamental_measured,diviate)
                partial_fqs = self.compute_partial(filtered, serial_no=i)
                self.differences.append(partial_fqs-i*self.fundamental_measured)
        if mode =='full': # after a rough estimation of beta as above, use beta iteratively to guess the next position of partial
            diviate_temp = diviate
            con=diviate
            old_beta = 0
            cnt = 0
            diviate = round(diviate/(self.sampling_rate/self.ff_t.size))
            for i in range(2,no_of_partials):
                if r is not None:
                   diviate_temp = con*math.sqrt(r*i+r**2)
                diviate = round(diviate_temp/(self.sampling_rate/self.ff_t.size))
                factor = 1
                if i>12:
                    k, d = zip(*self.differences.items())
                    beta = compute_beta(d=d,k=k,track=self,mode='full')
                    if abs(beta-old_beta)<0.01*old_beta:
                        cnt+=1
                        if cnt>1:
                            print(i)
                            #flag = True
                            break
                        #else: cnt = 0
                        #flag = True
                    #else:
                    #    flag = False
                    old_beta = beta
            #     print(beta)
                #  if beta>0:
                    factor = math.sqrt(1+i**2*beta)
                filtered = self.zero_out(i*factor*self.fundamental_measured,diviate)
                partial_fqs = self.compute_partial(filtered, serial_no=i)
                self.differences={}
                self.differences[i] = (partial_fqs-i*self.fundamental_measured) #assign to dictionary
        #no_of_partials = 50
        if mode=='barbancho':
            for i in range(2,no_of_partials):
                diviate = i*math.sqrt(1+beta*i**2)*r
                #diviate = 80
                diviate = round(diviate/(self.sampling_rate/self.ff_t.size))
                filtered = self.zero_out(i*math.sqrt(1+beta*i**2)*(cand_fund+d),diviate)
                partial_fqs = self.compute_partial(filtered, serial_no=i)
    
    def compute_partial(self, filtered, serial_no):
        partial = partials(serial_no,
                           expected_freq=self.fundamental*serial_no,
                           partial_fqs=None, ff_t=filtered,frequencies=self.frequencies)
        max_peak = partial.get_max_freqs()
        self.append_partial(max_peak)
        return max_peak

class partials(my_signals):
    
    def __init__(self, serial_no, expected_freq
            ,partial_fqs=None, ff_t=None, sampling_rate=44100,frequencies=None):
        self.frequencies = frequencies
        if partial_fqs is None:
            pass
        else:
            self.partial_fqs = partial_fqs
        if ff_t is None:
            pass
        else:
            self.ff_t = ff_t
        if serial_no is None:
            raise Exception('i need a serial Number!')
        else:
            self.serial_no = serial_no
        if expected_freq is None:
            pass
        else: 
            expected_freq = expected_freq
        self.sampling_rate = sampling_rate
        
def compute_least(u,y):
    def model(x, u):
        return x[0] * u**3 + x[1]*u + x[2]
    def fun(x, u, y):
        return model(x, u)-y
    def jac(x, u, y):
        J = np.empty((u.size, x.size))
        J[:, 0] = u**3
        J[:, 1] = u
        J[:, 2] = 1
        return J
    x0=[0.00001,0.00001,0.000001]
    res = least_squares(fun, x0, jac=jac,bounds=(0,np.inf), args=(u, y),loss = 'soft_l1', verbose=0)
    return res.x    
   
def compute_beta(y=None,d=None,k=None,track=None,mode='flat'): #least squares
    if mode=='flat':
        u=np.arange(2.0,y.shape[0]+2)
        res=compute_least(u,y)
        [a,b,c]=res
        beta=2*a/(track.fundamental_measured+b)
        return beta
    else:
        u=np.array(k)
        res=compute_least(u,d)
        [a,b,c]=res
        beta=2*a/(track.fundamental_measured+b)
        return beta

def average_peak(peak_weights, peak_freqs):
    summ = 0
    for index, x in enumerate(peak_weights):
        summ +=peak_weights[index]*peak_freqs[index]
    avg = summ/sum(peak_weights)
    return avg

def compute_fret(string,midi):
    string_dict = {'string0' : 40, 'string1' :  45, 'string2' : 50 ,
                'string3' : 55 , 'string4' :  59, 'string5' :64}
    return midi-string_dict[string]

def determine_string(track):
    string_dict = {'string0' : 40, 'string1' :  45, 'string2' : 50 ,
                'string3' : 55 , 'string4' :  59, 'string5' :64}
    for key in string_dict.keys():
        if key in track:
            return int(key[-1])
    


def determine_combinations(f_cand): # returns all posible ways to play a candidate fundamental
    ret = []
   # fret_range = [range(40,53),range(45,58),range(50,63),range(55,68),range(59,72),range(64,77)]
    fret_range = [range(40,57),range(45,52),range(50,67),range(55,72),range(59,76),range(64,79)]
    midi_f_cand = round(12*math.log(f_cand/440,2)+69)
    for index, x in enumerate(fret_range):
        if midi_f_cand in list(x):
            ret.append((midi_f_cand, index, compute_fret('string'+str(index), midi_f_cand)))
    return ret

def perfect_beta(midi,track):
    x = note_instance(name= track,midi_note=midi)
    x.compute_partials(50,x.fundamental_measured/2,mode ='full')
  #  print(list(zip(x.list_of_partials, [x.fundamental_measured*i for i in range(2,50)])))
   # print(x.fundamental_measured)
    k, d = zip(*x.differences.items())
    final_beta =compute_beta(y=None,d=d,k=k,track=x,mode='full')
    del x
    return final_beta