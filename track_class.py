
from initialize_workspace import *
import jams
import librosa
import numpy as np
from madmom.audio.filters import hz2midi
from madmom.evaluation.onsets import onset_evaluation, OnsetEvaluation
from madmom.features.onsets import CNNOnsetProcessor, OnsetPeakPickingProcessor
import crepe
import warnings
import pickle
from helper import note_instance, compute_beta
from genetic import genetic
from misc import read_correlate_matrix, compute_fret, get_probs, plot_confusion_matrix
from matplotlib import lines as mlines, pyplot as plt
#temp lbiraries not really needed
import random
import shutil
import torch # __greg__
from model import TCN # __greg__
from madmom.features.onsets import peak_picking, OnsetPeakPickingProcessor # __greg__
# from madmom.evaluation import onsets
# print(onsets)


# dataset = 'mix'
# dataset = 'hex'
dataset = 'hex_cln'
# dataset = 'mic'

# workspace = initialize_workspace('/media/estfa/10dcab7d-9e9c-4891-b237-8e2da4d5a8f2/data_2')
# workspace = initialize_workspace('./dataset') # __greg__
workspace = initialize_workspace('./dataset') # __greg__


HOP = 441 # 10 ms
# W_SIZE = 2048
W_SIZE = 1764
FS = 44100 
N_BANDS = 40


# coeff = read_correlate_matrix()
coeff = (1,1,1,1,1)
print(coeff)
c_est = sum(coeff)/len(coeff)
print(c_est)

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
        folder_len = name.rfind('/')
        temp_name = name[folder_len+1:-5]
        name = (workspace.workspace_folder+'/' +
                            dataset + '/' + temp_name + 
                                '_' + dataset + '.wav')

    elif mode == 'to_jams':
        folder_len = name.rfind('/')
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
    def __init__(self, jam_name, dataset): #, dataset
        self.jam_name = jam_name
        self.track_name = convert_name(jam_name, dataset, 'to_wav')
        self.true_tablature = self.read_tablature_from_jams() # NOTE:
        print(self.track_name)
        audio, sr = librosa.load(self.track_name, sr=FS, mono=False)
        
        self.audio = audio  # NOTE:
        self.sr = sr
        self.predicted_tablature = None
        self.rnn_tablature = None
        self.predicted_strings = None

        self.dur = librosa.get_duration(self.audio, sr=self.sr, n_fft=W_SIZE, hop_length=HOP)
        

        # print(len(self.audio.shape))

        self.Feats = []
        if len(self.audio.shape) > 1:  # mulitchannel
            for i in range(0,self.audio.shape[0]): 
                self.Feats += [ librosa.feature.melspectrogram(self.audio[i,:], sr=self.sr, n_mels=N_BANDS, n_fft=W_SIZE, hop_length=HOP) ] #_greg_
        else: # mono
            self.Feats += [ librosa.feature.melspectrogram(self.audio, sr=self.sr, n_mels=N_BANDS, n_fft=W_SIZE, hop_length=HOP) ] #_greg_
        
        # print(len(self.Feats))

        self.feats = self.Feats[0]
        # self.feats = librosa.feature.melspectrogram(self.audio, sr=self.sr, n_mels=N_BANDS, n_fft=W_SIZE, hop_length=HOP) #_greg_
        # # self.feats = librosa.cqt(self.audio, sr=self.sr, n_bins=N_BANDS, hop_length=HOP) #_greg_
        # if '01_BN2-131-B_solo' in jam_name:
            # print('CCCCCCCCCCCCCc', self.dur)
            # print()

    def predict_tablature(self, mode = 'FromAnnos'):
        '''mode --> {From_Annos, FromCNN} first reads 
        from annotations onset-pitch second estimates'''
        strings = []
        if mode == 'FromCNN':
            onsets, midi_notes = self.predict_notes_onsets()
            self.temp_tablature = Tablature(onsets, midi_notes, [])
            estim_tab, probs = self.rnn_predict_strings()
            temp = [(x[1],1) for x in estim_tab]
            self.rnn_tablature_FromCNN = Tablature(onsets, midi_notes, temp)
            fin_tab, gen = genetic(estim_tab, probs, coeff)
            for s in fin_tab:
                strings.append((s[1],1))
            self.predicted_tablature_FromCNN = Tablature(onsets, midi_notes, strings)

        elif mode == 'FromAnnos':
            self.temp_tablature = self.true_tablature
            estim_tab, probs = self.rnn_predict_strings()
            string_temp = [(x[1],max(p)) for x,p in zip(estim_tab,probs)]
            onset_temp = [(x[-2],1) for x in estim_tab]
            midi_temp = [(x[0],1) for x in estim_tab]
            self.rnn_tablature_FromAnnos = Tablature(onset_temp, midi_temp, string_temp)
            fin_tab, gen = genetic(estim_tab, probs, coeff)
            for s in fin_tab:
                strings.append((s[1],1))
            self.predicted_tablature_FromAnnos = Tablature(onset_temp, midi_temp, strings)
            
        elif mode == 'custom': #__greg__
            # audio_feats = librosa.feature.melspectrogram(audio_data_mix, sr=args.fs, n_mels=40, n_fft=args.w_size, hop_length=args.hop)
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
            annotations = x.return_onset_times()
            evalu = OnsetEvaluation(otimes, annotations, window=0.025)
            print("\n", evalu)
            self.predicted_tablature_FromTCN = otimes


    def read_tablature_from_jams(self):
        str_midi_dict = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}
        s = 0
        jam = jams.load(self.jam_name)
        tablature = []
        onsets = []
        midi_notes = []
        strings = []
        
        annos = jam.search(namespace='note_midi')
        if len(annos) == 0:
            annos = jam.search(namespace='pitch_midi')
        for string_tran in annos:
            for note in string_tran:
                start_time = note[0]
                end_time = start_time + 0.06
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

        return Tablature(onsets, midi_notes, strings)
    
    def predict_onsets(self):
        proc_0 = CNNOnsetProcessor()
        proc_1 = OnsetPeakPickingProcessor(threshold = 0.95,fps=100)
        predicts = proc_1(proc_0(self.track_name))

        #=====manually adding true onsets
        #predicts = [onset.prediction for onset in self.true_tablature.onsets]
        #====

        return list(zip(predicts, [1]*len(predicts))) # here correct it when i can get confidence

    def predict_notes_at_time(self,onsets):
        midi_notes = []

        local_onsets = list(zip(*onsets))[0]
        time, frequency, confidence, activation = crepe.predict(self.audio, self.sr, viterbi=True)
        time = list(time)
        frequency = list(frequency)
        confidence = list(confidence)
        for x in local_onsets:
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

    def predict_notes_onsets(self):
        onsets = self.predict_onsets()
        midi_notes = self.predict_notes_at_time(onsets)
        return onsets, midi_notes


    def return_onset_times(self): # __greg__
        tablature =  self.true_tablature
        trackOnsets = []
        # print('tablature:', tablature)
        for instance in zip(tablature.onsets, tablature.midi_notes, tablature.strings):    
            # print('AAAAAAAA')    
            onset, midi_note, string = instance
            # print(onset.prediction)
            trackOnsets.append(onset.prediction)
            # offset = onset.prediction + 0.06 # TODO: check again
            # start = int(round(onset.prediction*(self.sr)))
            # end = int(round(offset*(self.sr)))
            # instance_data = self.audio[start:end] #NOTE: use mono chanel
            # print(onset.prediction)

        return np.array(trackOnsets)


    def rnn_predict_strings(self):
    #initializations
        wrong_dict = {0:0,1:0,2:0,3:0,4:0,5:0}
        no_of_partials = 14
        probability_list = []
        estim_tab = []
        neigh_dict ={}

        #load models
        for midi in range(40,82):
            filename = workspace.model_folder + '/'+ str(midi) + '_rrn_model.sav'
            try:
                neigh_dict[midi] = pickle.load(open(filename, 'rb'))
            except FileNotFoundError:
                return 0

        #create and load track. 
        for instance in zip(self.temp_tablature.onsets,
                            self.temp_tablature.midi_notes, self.temp_tablature.strings):
            onset, midi_note, string = instance
            offset = onset.prediction + 0.06
            if 39<midi_note.prediction<82:
                start = int(round(onset.prediction*(self.sr)))
                end = int(round(offset*(self.sr)))
                if 'hex_cln' in self.track_name:
                    temp = self.audio[string.prediction,:]
                    instance_data = temp[start:end]
                else: # NOTE: mono chanel
                    instance_data = self.audio[start:end]

                #compute beta coeef
                x = note_instance(name = self.track_name, data = instance_data, midi_note = midi_note.prediction)
                x.compute_partials(no_of_partials,diviate=x.fundamental_measured/2)
                beta = compute_beta(y=np.array(x.differences),track=x)

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
                else:
                    estim_tab.append([midi_note.prediction,7, 
                                        7,onset.prediction, offset])
                    probability_list.append([7])

        return estim_tab, probability_list
    
    def get_accuracy_of_prediction(self, mode = 'FromAnnos'):
        if mode == 'FromAnnos':
            tab = self.rnn_tablature_FromAnnos
            rnn_tp = [onset.prediction for (onset, t_string, p_string) 
                                            in zip(self.true_tablature.onsets, 
                                                self.true_tablature.strings, tab.strings)
                                                    if t_string.prediction == p_string.prediction]
            
            inconclusive = [onset.prediction for (onset, t_string, p_string) 
                                in zip(self.true_tablature.onsets, 
                                    self.true_tablature.strings, tab.strings)
                                        if p_string.prediction == 7]

            print('accuracy of rnn is {} and inconclusive are {}'.format(len(rnn_tp)/tab.tab_len, len(inconclusive)/tab.tab_len))
            tab = self.predicted_tablature_FromAnnos
            ga_tp = [onset.prediction for (onset, t_string, p_string) 
                                            in zip(self.true_tablature.onsets, 
                                                self.true_tablature.strings, tab.strings)
                                                    if t_string.prediction == p_string.prediction]
            print('accuracy of ga is {}'.format(len(ga_tp)/tab.tab_len))
        elif mode == 'FromCNN':
            tab = self.predicted_tablature_FromCNN
            pre_on = [onset.prediction for onset in tab.onsets]
            tru_on = [onset.prediction for onset in self.true_tablature.onsets]
            tp, fp, tn, fn, errors = onset_evaluation(
                                                    pre_on, tru_on,
                                                        window = 0.025)
            recall = len(tp)/(len(tp)+len(fn))
            precision = len(tp)/(len(tp)+len(fp))
            f1_measure = 2*recall*precision/(recall+precision)
            print('onsets recall is {} and precision is {} and f1_measure is {}'.format(recall, precision, f1_measure))
            

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

    def __init__(self, onsets = [], midi_notes = [], strings = []):
        self.tab_len = len(onsets)
        self.onsets = [Onset(x) for x in onsets] # NOTE:
        self.midi_notes = [MidiNote(x) for x in midi_notes]
        self.strings = [String(x) for x in strings]
    
    def __str__(self):
        temp_str = ''
        for index, t in enumerate(zip(self.onsets,self.midi_notes,self.strings)):
            onset, midi, string = t
            fret = compute_fret(string.prediction,midi.prediction)
            temp_str += 'onset is {} midi is {} and string-fret {} {} \n'.format(
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

def compute_confusion(confusion_matrix, predicted_tablature, true_tablature, onset_window = 0.025, mode = 'ga'):
    for instance in zip(predicted_tablature.onsets,
                        predicted_tablature.midi_notes,
                            predicted_tablature.strings,
                            true_tablature.onsets, true_tablature.midi_notes,
                             true_tablature.strings):
        instance = [x.prediction for x in instance]
        p_onset, p_midi, p_string, t_onset, t_midi, t_string = instance
        if abs(t_onset - p_onset)<onset_window: # here add to propagate smallest window in case onset should be taken seriously
            if p_string == 7: # designated the inconclusive
                confusion_matrix[t_string][6] +=1
            elif p_midi == t_midi:
                confusion_matrix[t_string][p_string] +=1
            else:
                 raise Exception('The midi values are different something is wrong')
    return confusion_matrix


def get_features_and_targets(): # __greg__
    # jam_list = glob.glob(workspace.annotations_folder + '/single_notes/*solo*.jams')

    # Read list of test filenames
    with open('./names.txt', 'r') as f:
        TestSet = f.readlines()
    TestSet = [x.strip() for x in TestSet]

    # print(TestSet)
    path_to_store = './melspec_'+dataset+'/'

    try: shutil.rmtree(path_to_store)
    except: print('No need to clean old data...')

    try: os.mkdir(path_to_store)
    except: print('ERROR: Failed to create new data dir.'); exit(1)
    os.mkdir(path_to_store+'Train/')
    os.mkdir(path_to_store+'Test/')
    # except: shutil.rmtree(path_to_store)

    jam_list = glob.glob(workspace.annotations_folder + '/*.jams')
    for jam_name in jam_list:
        print(jam_name)
        # if not 'solo' in jam_name: 
        # if 'solo' in jam_name: 
        #     continue
        x = TrackInstance(jam_name, dataset)    

        # feats = x.feats
        Feats = x.Feats

        # get track onsets
        onset_times = x.return_onset_times()

        # print('onset_times:', onset_times[-1], 'feats:', feats.shape)

        # get baf i.e. frame ground truth
        onset_frames = librosa.core.time_to_frames(onset_times, sr=x.sr, hop_length=HOP)
        baf = np.array( [np.zeros( x.feats.shape[1] )]) # length
        baf[0, onset_frames] = 1.
        baf = np.swapaxes(baf, 0, 1)

        # print(baf.shape)
        # locations = np.nonzero(baf)[0]
        # print('AAAAAA', onset_frames[-1], locations[-1], onset_times[-1]) NOTE: locations != onset_times ...
        # print('AAAAAA', onset_frames[-10], locations[-10], onset_times[-10])

        filename = jam_name.split('/')[-1][:-5]

        for i, feats in enumerate(Feats):
            if len(Feats) == 1: 
                suffix='_'+dataset
            else:
                suffix='_'+dataset+'_ch_'+str(i)
                # print(suffix)
   
            if jam_name.split('/')[-1] in TestSet:
                print('test')
                np.savez(path_to_store+'Test/'+filename+suffix+'.npz', feats=feats, baf=baf, onset_times=onset_times)
            else:
                np.savez(path_to_store+'Train/'+filename+suffix+'.npz', feats=feats, baf=baf, onset_times=onset_times)

        # npzfile = np.load(path_to_store+filename+'.npz')

    return 0



def compute_confusion_matrixes():
    y_classes = ['E','A','D','G','B','e']
    x_classes = ['E','A','D','G','B','e', 'Inconclusive']
    correct_ga = 0
    correct_rnn = 0
    confusion_matrix_ga = np.zeros((6,6)) 
    confusion_matrix_rnn = np.zeros((6,7)) 
    jam_list = glob.glob(workspace.annotations_folder + '/single_notes/*solo*.jams')
    #jam_list = random.choices(glob.glob(workspace.annotations_folder + '/single_notes/*solo*.jams'), k = 3)
    for jam_name in jam_list:
    #jam_name = workspace.annotations_folder+'/05_BN1-147-Gb_solo.jams'
        x = TrackInstance(jam_name, dataset)
        x.predict_tablature('FromAnnos')
        x.get_accuracy_of_prediction('FromAnnos')
    #x.rnn_tablature_FromAnnos.tablaturize()
    #x.predicted_tablature_FromAnnos.tablaturize()
        confusion_matrix_ga = compute_confusion(confusion_matrix_ga, x.predicted_tablature_FromAnnos, x.true_tablature)
        confusion_matrix_rnn = compute_confusion(confusion_matrix_rnn, x.rnn_tablature_FromAnnos, x.true_tablature)
    for i in range(6):
        correct_ga += confusion_matrix_ga[i][i]
        correct_rnn += confusion_matrix_rnn[i][i]
    accuracy_ga = correct_ga/np.sum(confusion_matrix_ga)
    accuracy_rnn = correct_rnn/np.sum(confusion_matrix_rnn)
    print(confusion_matrix_ga)
    title = 'GA Confusion Matrix For ' + dataset + ' Recordings accuracy is ' + str(accuracy_ga)+'No Stop'
    plot_confusion_matrix(confusion_matrix_ga, y_classes, y_classes,
                            normalize = True, title = title)
    title = 'Rnn Confusion Matrix For ' + dataset + ' Recordings accuracy is ' + str(accuracy_rnn)+'No Stop'
    plot_confusion_matrix(confusion_matrix_rnn, x_classes, y_classes,
                            normalize = True, title = title)        


if __name__ == '__main__':
    get_features_and_targets() # __greg__
 
    # jam_name = workspace.annotations_folder+'/05_BN1-147-Gb_solo.jams'
    # x = TrackInstance(jam_name, dataset)
    # x.predict_tablature('custom')
    # print(x.predicted_tablature_FromTCN)

    # x.predict_tablature('FromAnnos')
    # x.rnn_tablature_FromAnnos.tablaturize()
    # x.predicted_tablature_FromAnnos.tablaturize()