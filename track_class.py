
from initialize_workspace import *
import jams
import librosa
import numpy as np
from madmom.audio.filters import hz2midi
from madmom.evaluation.onsets import onset_evaluation
from madmom.features.onsets import CNNOnsetProcessor, OnsetPeakPickingProcessor
import crepe
import warnings
import pickle
from helper import note_instance, compute_beta
from genetic import genetic
from misc import read_correlate_matrix, compute_fret, get_probs
from matplotlib import lines as mlines, pyplot as plt


dataset = 'mic'
workspace = initialize_workspace('/media/estfa/10dcab7d-9e9c-4891-b237-8e2da4d5a8f2/data_2')

coeff = read_correlate_matrix()
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
        temp_name = name[len(workspace.annotations_folder)+1:-5]
        name = (workspace.workspace_folder+'/' +
                            dataset + '/' + temp_name + 
                                '_' + dataset + '.wav')

    elif mode == 'to_jams':
        temp_name = name[len(workspace.annotations_folder)+1:len(dataset)-4]
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
        self.true_tablature = self.read_tablature_from_jams()
        audio, sr = librosa.load(self.track_name, sr=44100, mono=False)
        self.audio = audio
        self.sr = sr
        self.predicted_tablature = None
        self.rnn_tablature = None
        self.predicted_strings = None



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
                            self.temp_tablature.midi_notes):
            onset, midi_note = instance
            offset = onset.prediction + 0.06
            if 39<midi_note.prediction<82:
                start = int(round(onset.prediction*(self.sr)))
                end = int(round(offset*(self.sr)))
                #channel_data = data[string,:]
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
            print('accuracy of rnn is {}'.format(len(rnn_tp)/tab.tab_len))
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
        self.onsets = [Onset(x) for x in onsets]
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

jam_name = workspace.annotations_folder+'/05_BN1-147-Gb_solo.jams'
x = TrackInstance(jam_name, dataset)
x.predict_tablature('FromCNN')
x.get_accuracy_of_prediction('FromCNN')
x.rnn_tablature_FromCNN.tablaturize()
x.predicted_tablature_FromCNN.tablaturize()

