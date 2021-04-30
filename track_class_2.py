
from initialize_workspace import *
import jams
import math
import librosa
import soundfile as sf
import numpy as np
from midiutil import MIDIFile
from madmom import evaluation
from madmom.audio.filters import hz2midi, midi2hz
from madmom.evaluation.onsets import onset_evaluation, OnsetEvaluation
#from madmom.features.onsets import CNNOnsetProcessor, OnsetPeakPickingProcessor
import crepe
import glob
import warnings
import pickle
from helper import note_instance, compute_beta
from genetic import genetic
from misc import read_correlate_matrix, compute_fret, get_probs, plot_confusion_matrix, determine_combinations
from matplotlib import lines as mlines, pyplot as plt
#temp lbiraries not really needed
import random
import torch # __greg__
from model import TCN # __greg__
from madmom.features.onsets import peak_picking, OnsetPeakPickingProcessor
import madmom
from madmom.evaluation.onsets import onset_evaluation


dataset = 'mix'
workspace = initialize_workspace(os.path.join('C:\\','Users','stefa','Documents','guit_workspace'))

#coeff = read_correlate_matrix()
coeff = (1,1,1,1,1)
print(coeff)
c_est = sum(coeff)/len(coeff)
print(c_est)

classVars = [(0.00015010543699347213, 0.00010268685586965404),(0.00013722885303273712, 6.79217786897601*10**(-5)),
                (8.366353574780814*10**(-5), 5.388807411040291*10**(-5)),(4.33095691643002*10**(-5), 1.6316724151791738*10**(-5)),
                (5.6811205917038936*10**(-5), 2.807309660344347*10**(-5)), (3.055196481498139*10**(-5), 1.962542047757944*10**(-5))]
#classVars = [(0.00012046882688296577, 2.3483320044754826e-05), (0.0001093970333464828, 2.3483320044754826e-05),
#(8.181256798962631e-05, 2.3483320044754826e-05),(4.216788251707746e-05, 2.3483320044754826e-05),
#(5.6382752951044e-05, 1.3483320044754826e-05),(2.3084945090924958e-05, 0.2841556158908311e-05)]

#classVars = [(0.00010880844454611663, 4.224772579769999e-05), (5.338263876552266e-05, 2.553233568779919e-05),
#(3.559871619850302e-05, 5.137842837733525e-06), (1.968284410141703e-05, 3.180589616925209e-06),
#(5.496711668142129e-05, 4.382091165090362e-06), (1.934878162980727e-05, 1.0102876874148802e-06)]
classVars = [(0.00012937190004443645, 2.9734051438666772e-05),(4.75828998561585e-05, 2.6449410226449636e-05),
            (4.097933401107548e-05, 8.853208978536127e-06),(2.7617514386769436e-05, 7.853212227191108e-06),
            (3.976492803725843e-05, 1.6271577854956838e-05),(2.0775758832571158e-05, 3.145562499674651e-06)]
HOP = 441 # 10 ms
# W_SIZE = 2048
W_SIZE = 1764
FS = 44100
N_BANDS = 40

def compute_gauss(samples):
    samples_mean = np.median(samples)
    samples_std = np.std(samples)
    return samples_mean, samples_std


def get_betas():
    beta_dict = {}
    be3ta_dict = {}
    be12ta_dict = {}
    
    for string, midi in enumerate([40,45,50,55,59,64]):#[40,45,50,55,59,64]52,57,62,67,71,76, 40,45,50,55,59,64 [47,52,57,62,66,71]
        for fret in [0,3,12]:
        # string = 2
            betas = []
            track_list = glob.glob(os.path.join(workspace.workspace_folder,'crops', str(midi+fret), str(string),'good', '*.wav'))#
        #  track_list = random.choices(track_list, k = 100)
            for track in track_list[:500]:
                audio, sr = librosa.load(track, sr=44100, mono=False)
                start = 0# + 0.02
                end = int(round(0.06*(sr)))
                instance_data = audio[start:end]
                x = note_instance(name = track, data = instance_data, midi_note = (midi+fret))
                #x = note_instance(track, midi)
                x.compute_partials(no_of_partials = 14, diviate = x.fundamental_measured/2)
                beta = compute_beta(y=np.array(x.differences),track=x)
                print(track, beta)
                if beta>10**(-7):
                    betas.append(beta)
            if fret == 0:
                beta_dict[string] = compute_gauss(betas)[0]
            if fret == 3:
                be3ta_dict[string] = compute_gauss(betas)[0]
            if fret == 12:
                be12ta_dict[string] = compute_gauss(betas)[0]
    return beta_dict, be3ta_dict, be12ta_dict


def get_vars():
    classVars = []
    for string, midi in enumerate([40, 45, 50, 55, 59, 64]):#([40, 45, 50, 55, 59, 64])([47, 52,57, 62, 66, 69])
        cnt = 0
        samples = []
        track_list = random.choices(glob.glob(os.path.join(workspace.workspace_folder,'crops', str(midi), str(string), '*.wav')), k=30)
        for track in track_list:
            if cnt> 30:
                break
            else:
                x = note_instance(track, midi)
                x.compute_partials(no_of_partials = 14, diviate = x.fundamental_measured/2)
                beta = compute_beta(y=np.array(x.differences),track=x)
                if beta>10**(-7):
                    cnt+=1
                    samples.append(beta)
        classVars.append(compute_gauss(samples))
    return classVars

def compute_max_prob(classVars, sample, string_ar):
    probs = []
    for t in classVars:

        exponent =  math.exp((-(sample-t[0])**2)/(2*t[1]**2))
        print(exponent)
        probs.append((exponent )/(math.sqrt(2 * math.pi * t[1]**2)))
    #print(probs)
    #print(np.argmax(probs))
    return string_ar[np.argmax(probs)]

def convert_name(name, dataset = None, mode = 'to_wav'):
    '''modes --> {to_wav, to_jams}
    dataset --> {hex_cln, mic, mix}'''
    if dataset == None:
        if '.wav' in name:
            return name[:-4] + '.jams'
        elif '.jams' in name:
            return name[:-5] + '.wav'
        else:
            raise NameError('Not proper track extension neither wav or jams')
    elif mode == 'to_wav':
        folder_len = name.rfind('\\')
        temp_name = name[folder_len+1:-5]
        name = os.path.join(workspace.workspace_folder,
                            dataset, temp_name + 
                                '_' + dataset + '.wav')

    elif mode == 'to_jams':
        folder_len = name.rfind('\\')
        temp_name = name[folder_len+1:len(dataset)-4]
        name = workspace.annotations_folder + temp_name + '.jams'
    else:
        raise NameError('Not Proper Mode choose to_wav or to_jams')
    if mode == 'to_jams' and dataset not in name:
        return name
    elif mode == 'to_wav' and dataset in name:
        return name
    else:
        raise NameError('Validity check failed name to be returned is {}'.format(name))
    return name




class TrackInstance():
    def __init__(self, jam_name, dataset, AnnosMode = True): #, dataset
        if AnnosMode:
            self.jam_name = jam_name
            self.track_name = convert_name(jam_name, dataset, 'to_wav')
            self.true_tablature, _ = self.read_tablature_from_jams()
            print(self.track_name)
            audio, sr = librosa.load(self.track_name, sr=44100, mono=False)
            self.audio = audio
            self.sr = sr
            self.predicted_tablature = None
            self.rnn_tablature = None
            self.predicted_strings = None
            self.duration = librosa.get_duration(audio, sr)
            
        else:
            self.track_name = jam_name
            print(self.track_name)
            audio, sr = librosa.load(self.track_name, sr=44100, mono=False)
            self.audio = audio
            self.sr = sr
            self.predicted_tablature = None
            self.rnn_tablature = None
            self.predicted_strings = None
            self.duration = librosa.get_duration(audio, sr)
        self.Feats = []
        if len(self.audio.shape) > 1:  # mulitchannel
            for i in range(0,self.audio.shape[0]): 
                self.Feats += [ librosa.feature.melspectrogram(self.audio[i,:], sr=self.sr, n_mels=N_BANDS, n_fft=W_SIZE, hop_length=HOP) ] #_greg_
        else: # mono
            self.Feats += [ librosa.feature.melspectrogram(self.audio, sr=self.sr, n_mels=N_BANDS, n_fft=W_SIZE, hop_length=HOP) ] #_greg_
        
        # print(len(self.Feats))

        self.feats = self.Feats[0]
  

    def read_tablature_from_jams(self):
        str_midi_dict = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}
        s = 0
        jam = jams.load(self.jam_name)
        tablature = []
        onsets = []
        midi_notes = []
        strings = []
        offsets = []
        
        annos = jam.search(namespace='note_midi')
        if len(annos) == 0:
            annos = jam.search(namespace='pitch_midi')
        for string_tran in annos:
            for note in string_tran:
                start_time = note[0]
                end_time = start_time + note[1]
                midi_note = note[2]
                fret = int(round(midi_note - str_midi_dict[s]))
                string = s
                tablature.append([round(midi_note),string,fret,start_time,end_time])
            s += 1
        tablature.sort(key=lambda x: x[3])

        for instance in tablature:
            onsets.append((instance[3], 1))
            midi_notes.append((instance[0], 1))
            strings.append((instance[1], 1))
            offsets.append((instance[4],1))

        return Tablature(onsets, midi_notes, strings, offsets), tablature

    def predict_tablature(self, onset = 'FromAnnos', pitch = 'FromAnnos', method = 'rrn_model.sav'):
        '''mode --> {From_Annos, FromCNN} first reads 
        from annotations onset-pitch second estimates'''
        strings = []
        if onset == 'FromAnnos':
            onsets = [(x.prediction, 1) for x in self.true_tablature.onsets]
        elif onset == 'TCN':
            onsets = self.get_onsets()
        if pitch == 'FromAnnos':
            if onset == 'FromAnnos':
                midi_notes = [(x.prediction, 1) for x in self.true_tablature.midi_notes]
            else:
                midi_notes = self.match_midi_notes(onsets)
        elif pitch == 'Crepe':
            #temp_onsets = [(x[0]+0.01, 1) for x in onsets] # add 0.02 to onsets for midi
            midi_notes = self.predict_notes_at_time(onsets)

        self.temp_tablature = Tablature(onsets, midi_notes, [(1,1) for x in onsets])
        estim_tab, probs = self.rnn_predict_strings(method = method, mode = 'other')
        temp = [(x[1],1) for x in estim_tab]
        self.rnn_tablature = Tablature(onsets, midi_notes, temp)
        fin_tab, gen = genetic(estim_tab, probs, coeff)
        for s in fin_tab:
            strings.append((s[1],1))
        self.predicted_tablature = Tablature(onsets, midi_notes, strings)

    def match_midi_notes(self, onsets):
        midi_notes = []
        for onset in onsets:
            print(onset[0])
            p_onset, p_midi, p_string = match_onset(onset[0], self.true_tablature)
            print(p_onset, p_midi, p_string)
            if (p_onset, p_midi, p_string) != (1,1,1):
                midi_notes.append((p_midi,1))
        return midi_notes


    def get_onsets(self):
        ksize=3
        nhid=256
        levels=4
        dropout=0.25
        dilations=True

        # TODO add args as self.
        def run_final_test(feats, input_type, model=None):
            model_name = "./models/TCN_Audio_0.pt"
            model = torch.load(model_name, map_location='cpu')
            # print(model)
            with torch.no_grad():
                x = feats.transpose(0,1)
                output = model(x.unsqueeze(0))               
            return output

        audio_feats = self.feats
        audio_feats = torch.Tensor(audio_feats.astype(np.float64))
        n_audio_channels = [nhid] * levels # e.g. [150] * 4
        print('\nModel Running...')
        model = TCN(40, 2, n_audio_channels, ksize, dropout=dropout, dilations=dilations)
        output = run_final_test(audio_feats, 'Audio', model)
        # output = run_final_test(audio_feats, 'Audio')
        print('\nModel Reults ready!!')
        output = output.squeeze(0).cpu().detach()
        print('output', output.size())
        oframes = peak_picking(activations=output[:,0].numpy(), threshold=0.5, pre_max=2, post_max=2) # madmom method
        otimes = librosa.core.frames_to_time(oframes, sr=self.sr, hop_length=HOP)  
        annotations = self.return_onset_times()
        evalu = OnsetEvaluation(otimes, annotations, window=0.025)
        print("\n", evalu)
        
        onsets = [(x, 1) for x in otimes if (x+0.04)<self.duration]
        print(onsets)
        return onsets

    def return_onset_times(self): # __greg__
        tablature =  self.true_tablature
        trackOnsets = []
        for instance in zip(tablature.onsets, tablature.midi_notes, tablature.strings):        
            onset, midi_note, string = instance
            trackOnsets.append(onset.prediction)

        return np.array(trackOnsets)

    def predict_notes_at_time(self,onsets):
        midi_notes = []

        local_onsets = list(zip(*onsets))[0]
        time, frequency, confidence, activation = crepe.predict(self.audio, self.sr, viterbi=True)
        time = list(time)
        frequency = list(frequency)
        confidence = list(confidence)
        for x in local_onsets: # fix bug when rounding exceeds time indexing
            if (x+0.02) < self.duration:
                x = x+0.02
            else:
                print('exceeding duration ', self.duration, x)
            if round(x,2)*100>len(time):
                ind = len(time) - 20
            else:
                ind = time.index(round(x,2))
            f_ind = ind # check to get better predictions
            max_so_far = 0
            if ind != 0 and ind<len(local_onsets)-3:
                for p in range(ind-1,ind+2): # maybe can fix fundumental here, and not just round it
                    if confidence[p]>max_so_far:
                        f_ind = p
                        max_so_far = confidence[p]
            midi_notes.append((round(hz2midi(frequency[f_ind])), confidence[f_ind]))
        return midi_notes

    def rnn_predict_strings(self, mode = 'rnn', method = 'rrn_model.sav'):
    #initializations
        cnt_g, cnt_b = 0, 0
        
        wrong_dict = {0:0,1:0,2:0,3:0,4:0,5:0}
        no_of_partials = 14
        probability_list = []
        estim_tab = []
        neigh_dict ={}
        if method == 'probs':
            for instance in zip(self.temp_tablature.onsets,
                                    self.temp_tablature.midi_notes, self.temp_tablature.strings):
                onset, midi_note, string = instance
                offset = onset.prediction + 0.06
                if 39<midi_note.prediction<82:
                    start = int(round((onset.prediction+ 0.02)*(self.sr)))# + 0.02
                    end = int(round(offset*(self.sr)))
                    if 'hex_cln' in self.track_name:
                        temp = self.audio[string.prediction,:]
                        instance_data = temp[start:end]
                    else:
                        instance_data = self.audio[start:end]
                    
                    #compute beta coeef
                    x = note_instance(name = self.track_name, data = instance_data, midi_note = midi_note.prediction)
                    x.compute_partials(no_of_partials,diviate=x.fundamental_measured/2)
                    beta = compute_beta(y=np.array(x.differences),track=x)
                if beta>10**(-7):
                    combinations = determine_combinations(midi_note.prediction)
                    c = [(classVars[x[1]][0]*2**(x[2]/6), classVars[x[1]][1]*2**(x[2]/6)) for x in combinations]
                 #   print(c)
                    string_ar = [x[1] for x in combinations]
                    strPredict = compute_max_prob(c, beta, string_ar)
                   # print(strPredict)
                    estim_tab.append([midi_note.prediction,strPredict, 
                                        compute_fret(strPredict,midi_note.prediction),
                                            onset.prediction, offset])
                else:
                    estim_tab.append([midi_note.prediction,7, 
                                        7,onset.prediction, offset])
                    probability_list.append([7])
        elif method == 'exp_aph' or method == 'heu_nor' or method == 'heu_exp' or method == 'heu_lin':
            not_few = True
            if not_few:
                print('evaluationg on 100 instances')
                # here are 100 instances medians
                beta_dict = {0: 0.00012423557794641988, 1: 9.720145237288956*10**(-5), 2: 5.3187112359074767*10**(-5),
                                 3: 3.26859277852029*10**(-5), 4: 5.799282669545177*10**(-5), 5: 2.2414154979144663*10**(-5)}
                be12ta_dict = {0:0.0005, 1: 0.00031423736770725075, 2: 0.00019662386002230382, 3: 0.00011019271161000297, 4: 0.0002542429573483972, 5: 8.507061123139849*10**(-5)}
                #be7ta_dict = {0:0.00028067356406657405, 1:0.00019606672260039483, 2: 8.410550812676359*10**(-5), 3:5.9955167910525*10**(-5), 4:0.00014868485409272555, 5: 5.016285201711486*10**(-5)}
                # here are 300 instances medians
                #beta_dict = {0: 0.00013337059005140732, 1: 6.524607894682703*10**(-5), 2: 4.801508290962539*10**(-5), 3: 3.148704491571871*10**(-5), 4: 4.848633278275628*10**(-5), 5: 2.0054885729404285*10**(-5)}
                #be12ta_dict = {0: 0.00038904898639055, 1: 0.00032726987664676347, 2: 0.00019427715203739686, 3: 0.00010780362590427734, 4: 0.00025799506115364515, 5: 8.535150793509763*10**(-5)}
                #be7ta_dict = {0:0.00028067356406657405, 1:0.00019606672260039483, 2: 8.410550812676359*10**(-5), 3:5.9955167910525*10**(-5), 4:0.00014868485409272555, 5: 5.016285201711486*10**(-5)}
            
                be3ta_dict = {0: 0.0001816286632303884, 1: 0.00013950205156766453, 2: 7.464415296952934*10**(-5),
                             3: 3.733194867341091*10**(-5), 4: 9.22147592695891*10**(-5), 5: 3.2627247921250496*10**(-5)}
            
            else:
                beta_dict = {0 : 0.00014480986291923184, 1: 7.659973074846505*10**(-5), 2 : 5.8994854078247874*10**(-5) , 
                    3 : 2.552008128057158*10**(-5) , 4 : 4.5259637662927454*10**(-5) , 5 : 1.9121304067161012*10**(-5)} #median dict
                print(beta_dict)
                be3ta_dict = {0 : 0.0001866935593848972, 1: 0.000133476181236889, 2 : 7.081690874004888*10**(-5) , 
                3 : 4.006386128439038*10**(-5), 4 : 8.720242932367596*10**(-5) , 5 : 3.362048511305764*10**(-5)}
                be7ta_dict = {0:0.0003096405354500053, 1:0.00023354396264995843, 2: 8.744114789848388*10**(-5), 3:5.3086622113959224*10**(-5), 4:0.00014763598813301353, 5: 6.016285201711486*10**(-5)}
                be12ta_dict = {0: 0.00038904898639055, 1:0.0003842323300012583, 2: 0.00020038953613093642, 3:0.00012063838866656948, 4:0.0002730013954186958, 5: 8.054734638103516*10**(-5)}
           # beta_dict, be3ta_dict, be12ta_dict = get_betas()
            #print(beta_dict, be3ta_dict, be12ta_dict)
            #be12ta_dict[0] = 0.0005
            print(beta_dict, be3ta_dict, be12ta_dict)
            for instance in zip(self.temp_tablature.onsets,
                                self.temp_tablature.midi_notes, self.temp_tablature.strings):
                onset, midi_note, string = instance
                offset = onset.prediction + 0.06
                if offset < self.duration:
                    if 39<midi_note.prediction<82:
                        start = int(round((onset.prediction)*(self.sr)))# + 0.02
                        end = int(round(offset*(self.sr)))
                        if 'hex_cln' in self.track_name:
                            temp = self.audio[string.prediction,:]
                            instance_data = temp[start:end]
                        else:
                            instance_data = self.audio[start:end]

                        #compute beta coeef
                        x = note_instance(name = self.track_name, data = instance_data, midi_note = midi_note.prediction)
                        x.compute_partials(no_of_partials,diviate=x.fundamental_measured/2)
                        beta = compute_beta(y=np.array(x.differences),track=x)
                        #print(beta)
                        if beta>10**(-7):
                            combinations = determine_combinations(midi_note.prediction)
                            min_so_far = 1
                            for comb in combinations:
                                if method == 'exp_aph':
                                #    a = 6/5 * (math.log2(be12ta_dict[comb[1]]) - math.log2( be7ta_dict[comb[1]]))
                                #    c = (math.log2(be12ta_dict[comb[1]]) - math.log2( beta_dict[comb[1]]))*6 - 12*a
                                    
                                    a = (6/9)* (math.log2(be12ta_dict[comb[1]]) - math.log2( be3ta_dict[comb[1]]))
                                    c = (math.log2(be12ta_dict[comb[1]]) - math.log2( beta_dict[comb[1]]))*6 - 12*a
                                    temp_beta = beta_dict[comb[1]]*2**((a*comb[2]+c)/6)
                                if method == 'heu_nor':
                                    temp_beta = beta_dict[comb[1]]*2**(comb[2]/6)
                                elif method == 'heu_exp':
                                    a = (math.log2(be12ta_dict[comb[1]]) - math.log2( beta_dict[comb[1]]))/2
                                    temp_beta = beta_dict[comb[1]]*2**(a*comb[2]/6)
                                elif method == 'heu_lin':
                                    d = 6*(math.log2(be12ta_dict[comb[1]]) - math.log2( beta_dict[comb[1]])) - 12
                                    temp_beta = beta_dict[comb[1]]*2**((comb[2]+d)/6)
                                elif method == 'exp_sec':
                                    d = (math.log2(be12ta_dict[comb[1]]) - math.log2( beta_dict[comb[1]]))*6 -12
                                    temp_beta = beta_dict[comb[1]]*2**((d+comb[2])/6)
                                if abs(temp_beta-beta)<min_so_far:
                                    min_so_far = abs(temp_beta-beta)
                                    string = comb[1]
                            estim_tab.append([midi_note.prediction,string, 
                                                compute_fret(string,midi_note.prediction),
                                                    onset.prediction, offset])
                            probability_list.append(1)
                        else:
                            estim_tab.append([midi_note.prediction,7, 
                                                7,onset.prediction, offset])
                            probability_list.append([7])
        else:
            

            #load models rnn
            if mode == 'rnn':
                for midi in range(40,82):
                    filename = os.path.join(workspace.model_folder, str(midi) + method)
                    try:
                        neigh_dict[midi] = pickle.load(open(filename, 'rb'))
                    except FileNotFoundError:
                        return 0
            elif mode == 'epiphone':
                for midi in range(40,82):
                    filename = os.path.join(workspace.model_folder, str(midi) + '_epiphone_gnb_model.sav')
                    try:
                        neigh_dict[midi] = pickle.load(open(filename, 'rb'))
                    except FileNotFoundError:
                        return 0
            elif mode == 'aristotle':
                for midi in range(40,82):
                    filename = os.path.join(workspace.model_folder, str(midi) + '_aristotle_gnb_model.sav')
                    try:
                        neigh_dict[midi] = pickle.load(open(filename, 'rb'))
                    except FileNotFoundError:
                        return 0
            #load models bayes
            else:
                for midi in range(40,82):
                    filename = os.path.join(workspace.model_folder, str(midi) + method) #gnb
                    print(filename)
                    try:
                        neigh_dict[midi] = pickle.load(open(filename, 'rb'))
                    except FileNotFoundError:
                        return 0

            #create and load track. 
            print(self.temp_tablature.tab_len)
            for instance in zip(self.temp_tablature.onsets,
                                self.temp_tablature.midi_notes, self.temp_tablature.strings):
                onset, midi_note, string = instance
                offset = onset.prediction + 0.06
                if 39<midi_note.prediction<82:
                    start = int(round((onset.prediction + 0.02)*(self.sr)))
                    end = int(round(offset*(self.sr)))
                    if 'hex_cln' in self.track_name:
                        temp = self.audio[string.prediction,:]
                        instance_data = temp[start:end]
                    else:
                        instance_data = self.audio[start:end]

                    #compute beta coeef
                    x = note_instance(name = self.track_name, data = instance_data, midi_note = midi_note.prediction)
                    x.compute_partials(no_of_partials,diviate=x.fundamental_measured/2)
                    beta = compute_beta(y=np.array(x.differences),track=x)

                    print('beta is {} and midi is {}'.format(beta, midi_note.prediction))

                    #test
                    if 0.0000001<beta<0.001:
                        with warnings.catch_warnings(record=True) as w:
                            warnings.simplefilter("always")
                            prediction = neigh_dict[midi_note.prediction].predict([[beta]])[0]
                            [prob] = neigh_dict[midi_note.prediction].predict_proba([[beta]])
                            estim_tab.append([midi_note.prediction,prediction, 
                                                compute_fret(prediction,midi_note.prediction),
                                                    onset.prediction, offset])

                            probability_list.append(get_probs(prob, 
                                                    neigh_dict[midi_note.prediction].classes_))

                            cnt_g += 1
                            #sf.write(os.path.join(workspace.workspace_folder,'good',str(cnt_g) + 'good'+str(midi_note.prediction) +'.wav'), instance_data, self.sr, 'PCM_24')
                    else:
                        estim_tab.append([midi_note.prediction,7, 
                                            7,onset.prediction, offset])
                        probability_list.append([7])
                        cnt_b += 1
                        #sf.write(os.path.join(workspace.workspace_folder,'bad', str(cnt_b) + 'bad'+str(midi_note.prediction) +'.wav'), instance_data, self.sr, 'PCM_24')
                else: print('midi out of bounds')
        return estim_tab, probability_list

class Predictions():
    def __init__(self, tup):
        prediction, confidence = tup
        self.prediction = prediction
        self.confidence = confidence
        return None

class Onset(Predictions):
    def some_func():
        pass

class MidiNote(Predictions):
    def some_func():
        pass

class String(Predictions):
    def some_func():
        pass

class Tablature():

    def __init__(self, onsets = [], midi_notes = [], strings = [], offsets = []):
        self.tab_len = len(onsets)
        self.onsets = [Onset(x) for x in onsets]
        self.midi_notes = [MidiNote(x) for x in midi_notes]
        self.strings = [String(x) for x in strings]
        self.offsets = [Onset(x) for x in offsets]
    
    def __str__(self):
        temp_str = ''
        for index, t in enumerate(zip(self.onsets,self.midi_notes,self.strings)):
            onset, midi, string = t
            fret = compute_fret(string.prediction,midi.prediction)
            temp_str += 'onset is {} midi is {} and string-fret {} {} /n'.format(
                                                                    onset.prediction,
                                                                     midi.prediction, 
                                                                     string.prediction, fret)
        return temp_str

    def get_tablature_as_list(self):
        for b in zip(self.onsets, self.midi_notes, self.strings):
            tab.append(b[1],b[2], compute_fret(b[2],b[1]), b[0])
 
    def tablaturize(self):
        str_midi_dict = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}
        string_dict = {0: 'E', 1: 'A', 2: 'D', 3: 'G', 4: 'B', 5: 'e'}
        style_dict = {0 : 'r', 1 : 'y', 2 : 'b', 3 : '#FF7F50', 4 : 'g', 5 : '#800080', 7 : 'magenta'}
        handle_list = []
        plt.figure(figsize=(12, 4))
        for k in range(6):
            handle_list.append(mlines.Line2D([], [], color=style_dict[k],
                                            label=string_dict[k]))
        for index, t in enumerate(zip(self.onsets,self.midi_notes,self.strings)):
            onset, midi, string = t
            fret = compute_fret(string.prediction,midi.prediction)
            plt.scatter(index, string.prediction, marker="${}$".format(fret), color =
                style_dict[string.prediction]) #onset.prediction
    #  plt.xlabel('Time (sec)')
        plt.xlabel('Time (As Index)')
        plt.ylabel('String Number')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
                handles=handle_list, ncol=8)
        plt.show()

    def output_to_midi(self, tempo = 80):
        track    = 0
        channel  = 0
        volume   = 100
        MyMIDI = MIDIFile(6)  # One track, defaults to format 1 (tempo track is created automatically)
        time     = 0
        MyMIDI.addTempo(track, time, tempo)
        duration = 1/8
        for index, instance in enumerate(zip(self.onsets,self.midi_notes,self.strings)):
            onset, midi, string = instance
            onset_time = onset.prediction*tempo/60
            print(string.prediction, index)
            MyMIDI.addNote(string.prediction, channel, midi.prediction, onset_time, duration, volume)
        
        
        with open("epiphone.mid", "wb") as output_file:
            MyMIDI.writeFile(output_file)


def match_onset(onset, tablature):
    for instance in zip(tablature.onsets,
                        tablature.midi_notes,
                            tablature.strings):
        instance = [x.prediction for x in instance]
        p_onset, p_midi, p_string = instance
        if abs(onset - p_onset) < 0.025:
            return p_onset, p_midi, p_string
    print('onset not found')
    return (1,1,1) # onset not found

def compute_confusion_2(confusion_matrix, predicted_tablature, true_tablature, onset_window = 0.025, mode = 'ga', true_onset = 0):
    # fix onsets checking before. Keep only the true tablature onsets, and designate wrong onset in case of difference
    false_negative = 0
    for instance in zip(true_tablature.onsets, true_tablature.midi_notes,
                             true_tablature.strings):
        instance = [x.prediction for x in instance]
        t_onset, t_midi, t_string = instance
        p_onset, p_midi, p_string = match_onset(t_onset, predicted_tablature)
        if (p_onset, p_midi, p_string) != (1,1,1):
            true_onset += 1
      #  if abs(t_onset - p_onset)<onset_window: # here add to propagate smallest window in case onset should be taken seriously
            if p_midi != t_midi:
                confusion_matrix[t_string][6] +=1
            elif p_string == 7: # designated the inconclusive
                confusion_matrix[t_string][8] +=1
            elif p_midi == t_midi:
                confusion_matrix[t_string][p_string] +=1
        else:
            confusion_matrix[t_string][7] += 1
            false_negative += 1
            print('false negatives are ', false_negative)

    detections = [x.prediction for x in predicted_tablature.onsets]
    annotations = [x.prediction for x in true_tablature.onsets]
    tp, fp, tn, fn, _ =  onset_evaluation(detections, annotations, window=0.025)
    print(tp, fp, tn, fn)
    return confusion_matrix, true_onset, len(tp), len(fp), len(tn), len(fn)

def f_measure_calculator(predicted_tablature, true_tablature):
    pass

def compute_confusion(confusion_matrix, predicted_tablature, true_tablature, onset_window = 0.025, mode = 'ga'):   
    for instance in zip(predicted_tablature.onsets,
                        predicted_tablature.midi_notes,
                            predicted_tablature.strings,
                            true_tablature.onsets, true_tablature.midi_notes,
                             true_tablature.strings):
        instance = [x.prediction for x in instance]
        p_onset, p_midi, p_string, t_onset, t_midi, t_string = instance
        if abs(t_onset - p_onset)<onset_window: # here add to propagate smallest window in case onset should be taken seriously
            if p_midi != t_midi:
                confusion_matrix[t_string][6] +=1
            elif p_string == 7: # designated the inconclusive
                confusion_matrix[t_string][8] +=1
            elif p_midi == t_midi:
                confusion_matrix[t_string][p_string] +=1
        
    return confusion_matrix

def predict_some():
    jam_list = glob.glob(os.path.join(workspace.annotations_folder, 'single_notes','*solo*.jams'))
    for index, jam_name in enumerate(jam_list):
                print('we are at {} % '.format(index/len(jam_list)*100))
                x = TrackInstance(jam_name, dataset)
                
def compute_confusion_matrixes(md = 'FromAnnos', OnsetMethod = 'TCN'):
    y_classes = ['E','A','D','G','B','e']
    
    x_classes = ['E','A','D','G','B','e', 'wrong_pitch','false_neg_ons','Inconclusive']#,'false_neg_ons'
    note_count = 0
    jam_list = glob.glob(os.path.join(workspace.annotations_folder, 'single_notes','*solo*.jams'))
    for dataset in ['mix']:
        '''if dataset == 'mix':
            md ==  'Crepe'
            OnsetMethod = 'TCN'''
        if md ==  'FromAnnos' and OnsetMethod ==  'FromAnnos':
            methods = [ 'exp_aph','heu_lin','heu_exp','heu_nor'] #'heu_nor', 'exp_aph','heu_exp','mic',
        else:
            methods = ['exp_aph','heu_lin','heu_exp','heu_nor']
        #jam_list = random.choices(glob.glob(workspace.annotations_folder + '/single_notes/*solo*.jams'), k = 15)
        for method in methods: #, 'heu_lin''heu_exp','exp_aph','heu_exp','exp_aph','probs','mean_std_gnb_model.sav','real_trained_gnb_model.sav',,'rrn_model.sav','exp_rrn_model.sav','linear_rrn_model.sav'
            correct_ga = 0
            correct_rnn = 0
            tp, fp, tn, fn = 0, 0, 0, 0
            true_onset = 0
            confusion_matrix_ga = np.zeros((6,8))
            confusion_matrix_rnn = np.zeros((6,9))
            for index, jam_name in enumerate(jam_list):
                print('we are at {} % '.format(index/len(jam_list)*100))
                x = TrackInstance(jam_name, dataset)
                note_count += x.true_tablature.tab_len
                
                x.predict_tablature(onset = OnsetMethod, pitch = md, method = method)
                if md == 'Crepe':
                    pass
                if  OnsetMethod == 'TCN':
                    confusion_matrix_ga, _ ,tpp, fpp, tnp, fnp= compute_confusion_2(confusion_matrix_ga, x.predicted_tablature, x.true_tablature, true_onset = true_onset)
                    confusion_matrix_rnn, true_onset, tpp, fpp, tnp, fnp = compute_confusion_2(confusion_matrix_rnn, x.rnn_tablature, x.true_tablature,  true_onset = true_onset)
                    tp += tpp
                    fp += fpp
                    tn += tnp
                    fn += fnp
                    print('recall is ' + str(tp/(tp+fn)) + 'precision is ' + str(tp/(tp+fp)))
                else:
                    confusion_matrix_ga = compute_confusion(confusion_matrix_ga, x.predicted_tablature, x.true_tablature)
                    confusion_matrix_rnn = compute_confusion(confusion_matrix_rnn, x.rnn_tablature, x.true_tablature)
                
                correct_ga = np.trace(confusion_matrix_ga)
                correct_rnn = np.trace(confusion_matrix_rnn)
                print(confusion_matrix_rnn)
                print(confusion_matrix_ga)
                print(note_count, np.sum(confusion_matrix_ga))
                accuracy_ga = round(correct_ga/np.sum(confusion_matrix_ga), 3)
                accuracy_rnn = round(correct_rnn/np.sum(confusion_matrix_rnn), 3)
                print('ga accuracy ', accuracy_ga)
                print('rnn accuracy ', accuracy_rnn)
            print(confusion_matrix_ga[:,6])
            wr_pitch = np.sum(confusion_matrix_ga[:,6])
            wr_onset = np.sum(confusion_matrix_ga[:,7])
            print(wr_pitch)
            tot = np.sum(confusion_matrix_ga) - wr_pitch - wr_onset
            tdr_ga = round(correct_ga/tot,3)
            tdr_rnn = round(correct_rnn/tot,3)
            print(tdr_ga)
            accuracy_ga = round(correct_ga/np.sum(confusion_matrix_ga), 3)
            accuracy_rnn = round(correct_rnn/np.sum(confusion_matrix_rnn), 3)
            print(confusion_matrix_rnn)
            print(confusion_matrix_ga)
            if OnsetMethod == 'TCN':
                tdrosga = round(correct_ga/true_onset,3)
                print(true_onset, correct_ga)
                tdrosrnn = round(correct_rnn/true_onset,3)
                tdropga = round(tot/true_onset,3)
                tdrpsga = round(correct_ga/tot,3)
                title = 'Perfect_GA_' + method + dataset + ' accuracy_is ' + str(accuracy_ga)+'TDROS_is ' + str(tdrosga) + 'TDROP_is ' + str(tdropga) + 'TDRPS_is ' + str(tdrpsga)
                plot_confusion_matrix(confusion_matrix_ga, x_classes[:6], y_classes,
                                        normalize = True, title = title)
                title = 'Perfect_rnn_' + method +  dataset + ' accuracy ' + str(accuracy_rnn) +'TDROS ' + str(tdrosrnn) + 'recall is ' + str(tp/(tp+fn)) + 'precision is ' + str(tp/(tp+fp))
                plot_confusion_matrix(confusion_matrix_rnn, x_classes, y_classes,
                                        normalize = True, title = title)
            else:
                title = 'Perfect_GA_0.02' + method + dataset + ' accuracy_is ' + str(accuracy_ga)+'TDR_is ' + str(tdr_ga)
                plot_confusion_matrix(confusion_matrix_ga, x_classes[:6], y_classes,
                                        normalize = True, title = title)
                title = 'Perfect_rnn_0.02' + method +  dataset + ' accuracy_is ' + str(accuracy_rnn) +'TDR_is ' + str(tdr_rnn)
                plot_confusion_matrix(confusion_matrix_rnn, x_classes, y_classes,
                                        normalize = True, title = title)

def train_guitar():
    no_of_partials = 14
    jam_list = glob.glob(os.path.join(workspace.annotations_folder, 'single_notes','*solo*.jams'))
    #jam_list =random.choices(glob.glob(workspace.annotations_folder + '/single_notes/*solo*.jams'), k = len(jam_list))
    betas_list = [[],[],[],[],[],[]]
    cnt = 0
    string_list = [ True, True,True, True,True, True]
    for jam_name in jam_list:
        x = TrackInstance(jam_name, dataset)
        for instance in zip(x.true_tablature.onsets, x.true_tablature.midi_notes,
                             x.true_tablature.strings):
            if cnt == 6:
                break
            onset, midi_note, string = instance
            offset = onset.prediction + 0.06
            if 39<midi_note.prediction<82:
                if string_list[string.prediction] == True:
                    start = int(round((onset.prediction + 0.02)*(x.sr)))
                    end = int(round(offset*(x.sr)))
                    if 'hex_cln' in x.track_name:
                        temp = x.audio[string.prediction,:]
                        instance_data = temp[start:end]
                    else:
                        instance_data = x.audio[start:end]

                    #compute beta coeef
                    y = note_instance(name = x.track_name, data = instance_data, midi_note = midi_note.prediction)
                    y.compute_partials(no_of_partials,diviate=y.fundamental_measured/2)
                    beta = compute_beta(y=np.array(y.differences),track=y)
                    if beta>10**(-7):
                        betas_list[string.prediction].append((beta, midi_note.prediction))
                        if len(betas_list[string.prediction])>5:
                            string_list[string.prediction]= False
                            cnt+=1
    #print(betas_list)
    return betas_list
'''(0.00010880844454611663, 4.224772579769999e-05)
(5.338263876552266e-05, 2.553233568779919e-05)
(3.559871619850302e-05, 5.137842837733525e-06)
(1.968284410141703e-05, 3.180589616925209e-06)
(5.496711668142129e-05, 4.382091165090362e-06)
(1.934878162980727e-05, 1.0102876874148802e-05)'''
def get_other_mean_var(betas_list):
    base_beta = [[],[],[],[],[],[]]
    res_list = []
    for index, beta_list in enumerate(betas_list):
        for tup in beta_list:
            fret = compute_fret(index, tup[1])
            base_beta[index].append(tup[0]/(2**(fret/6)))
    for b in base_beta:
        res_list.append(compute_gauss(b))
    print(res_list)
    return res_list



def teset_onset():
    jam_list = glob.glob(os.path.join(workspace.annotations_folder, 'single_notes','*solo*.jams'))
    tps = 0
    fns = 0
    fps = 0
    EvalObjects = []
    for index, jam_name in enumerate(jam_list):
        print('we are at {} % '.format(index/len(jam_list)*100))
        x = TrackInstance(jam_name, dataset)
        #note_count += x.true_tablature.tab_len
        onsets = x.get_onsets()
        onsets = [t[0] for t in onsets]
        print(onsets)
        tru_onsets = [t.prediction for t in x.true_tablature.onsets]
        eval_object = evaluation.onsets.OnsetEvaluation(onsets, tru_onsets, window=0.025)
        EvalObjects.append(eval_object)
    evalu = str(evaluation.onsets.OnsetMeanEvaluation(EvalObjects)).split()
    precision, recall, F_measure = evalu[13], evalu[15], evalu[17]
    print("Precision "+str(precision)+"\tRecall "+str(recall)+str("\tF_measure ")+str(F_measure))
            #tp, fp, tn, fn, _ =  onset_evaluation(onsets, tru_onsets, window=0.025)
            #tps += len(tp)
           # fps +=len(fp)
           # fns += len(fn)
  #  recall = tps/(tps+fns)
  #  precision = tps/(tps+fps)
  #  f_measure = 2*recall*precision/(recall+precision)
    print('recall is ',tps/(tps+fns),'precision is ', tps/(tps+fps), f_measure)
#eval_object = evaluation.onsets.OnsetEvaluation(predictions, annotations, window=0.025)
if __name__ == "__main__":
  #  betas_list = train_guitar()
  #  get_other_mean_var(betas_list)
    #get_vars()
   # for i in range(10):
   #     betas_list = train_guitar()
   #     classVars = get_other_mean_var(betas_list)
        #classVars = get_vars()
    #    print(classVars)
   # teset_onset()
    for s in ['FromAnnos','Crepe']:# 'FromAnnos',,'FromAnnos','FromAnnos','FromAnnos','FromAnnos',
        for b in ['FromAnnos','TCN']:
            if b == 'TCN' and s == 'FromAnnos':
               break
            compute_confusion_matrixes(s, OnsetMethod=b)
    '''jam_name = os.path.join(workspace.annotations_folder, 'single_notes','05_BN1-147-Gb_solo.jams')#'/05_BN1-147-Gb_solo.jams'
    x = TrackInstance(jam_name, dataset)
    x.predict_tablature(onset = 'TCN', pitch = 'Crepe')'''