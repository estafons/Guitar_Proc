
from initialize_workspace import *
import jams
import librosa
import soundfile as sf
import numpy as np
from midiutil import MIDIFile
from madmom.audio.filters import hz2midi, midi2hz
#from madmom.evaluation.onsets import onset_evaluation
#from madmom.features.onsets import CNNOnsetProcessor, OnsetPeakPickingProcessor
import crepe
import glob
import warnings
import pickle
from helper import note_instance, compute_beta
from genetic import genetic
from misc import read_correlate_matrix, compute_fret, get_probs, plot_confusion_matrix
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
            self.true_tablature = self.read_tablature_from_jams()
            print(self.track_name)
            audio, sr = librosa.load(self.track_name, sr=44100, mono=False)
            self.audio = audio
            self.sr = sr
            self.predicted_tablature = None
            self.rnn_tablature = None
            self.predicted_strings = None
            
        else:
            self.track_name = jam_name
            print(self.track_name)
            audio, sr = librosa.load(self.track_name, sr=44100, mono=False)
            self.audio = audio
            self.sr = sr
            self.predicted_tablature = None
            self.rnn_tablature = None
            self.predicted_strings = None
        self.feats = librosa.feature.melspectrogram(self.audio, sr=self.sr, n_mels=40, n_fft=2048, hop_length=480)     



    def predict_tablature(self, mode = 'FromAnnos'):
        '''mode --> {From_Annos, FromCNN} first reads 
        from annotations onset-pitch second estimates'''
        strings = []
        if mode == 'FromCNN':
            onsets, midi_notes = self.predict_notes_onsets()
            print('time of track is ', librosa.get_duration(self.audio, self.sr))
            self.temp_tablature = Tablature(onsets, midi_notes, [(1,1) for x in onsets])
            estim_tab, probs = self.rnn_predict_strings()
            temp = [(x[1],1) for x in estim_tab]
            self.rnn_tablature_FromCNN = Tablature(onsets, midi_notes, temp)
            fin_tab, gen = genetic(estim_tab, probs, coeff)
            for s in fin_tab:
                strings.append((s[1],1))
            self.predicted_tablature_FromCNN = Tablature(onsets, midi_notes, strings)
        elif mode == 'OnsetAnnos':
            onsets, midi_notes = self.predict_notes_onsets()
            onsets = [(x.prediction, 1) for x in self.true_tablature.onsets]
            #print('time of track is ', librosa.get_duration(self.audio, self.sr))
            self.temp_tablature = Tablature(onsets, midi_notes, [(1,1) for x in onsets])
            estim_tab, probs = self.rnn_predict_strings()
            temp = [(x[1],1) for x in estim_tab]
            self.rnn_tablature_FromCNN = Tablature(onsets, midi_notes, temp)
            fin_tab, gen = genetic(estim_tab, probs, coeff)

            for s in fin_tab:
                strings.append((s[1],1))
            self.predicted_tablature_OnsetAnnos = Tablature(onsets, midi_notes, strings)
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

    def get_onsets(self):
    
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
        print('Model Running...')
        model = TCN(40, 2, n_audio_channels, ksize, dropout=dropout, dilations=dilations)
        output = run_final_test(audio_feats, 'Audio', model)
        # output = run_final_test(audio_feats, 'Audio')
        print('Model Reults ready!!')
        output = output.squeeze(0).cpu().detach()
        print('output', output.size())
        oframes = peak_picking(activations=output[:,0].numpy(), threshold=0.5, pre_max=2, post_max=2) # madmom method
        otimes = librosa.core.frames_to_time(oframes, sr=self.sr, n_fft=2048, hop_length=480)
        end_time = librosa.get_duration(self.audio, self.sr) 
        onsets = [(round(x,2), 1) for x  in otimes if x<(end_time-0.05)] # problem with out of time onsets
      #  print(onsets)
        self.predicted_tablature_FromTCN = Tablature(onsets, [(1,1) for x in onsets], [(1,1) for x in onsets])
        return onsets



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
                start_time = note[0] + 0.02
                end_time = start_time + 0.04
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
     #   proc_0 = CNNOnsetProcessor()
     #   proc_1 = OnsetPeakPickingProcessor(threshold = 0.95,fps=100)
      #  predicts = proc_1(proc_0(self.track_name))

        #=====manually adding true onsets
        predicts = [onset.prediction for onset in self.true_tablature.onsets]
        #====

        return list(zip(predicts, [1]*len(predicts))) # here correct it when i can get confidence

    def predict_notes_at_time(self,onsets):
        midi_notes = []

        local_onsets = list(zip(*onsets))[0]
        time, frequency, confidence, activation = crepe.predict(self.audio, self.sr, viterbi=True)
        time = list(time)
        frequency = list(frequency)
        confidence = list(confidence)
        for x in local_onsets: # fix bug when rounding exceeds time indexing
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

    def predict_notes_onsets(self):
        onsets = self.predict_onsets() #here for madmom here change from true tab to tcn
        #onsets = self.get_onsets()
        print(onsets)
       # onsets = [(x.prediction, 1) for x in self.true_tablature.onsets] # test only pitch detection
        midi_notes = self.predict_notes_at_time(onsets)
        return onsets, midi_notes


    def rnn_predict_strings(self, mode = 'gnb'):
    #initializations
        cnt_g, cnt_b = 0, 0
        
        wrong_dict = {0:0,1:0,2:0,3:0,4:0,5:0}
        no_of_partials = 14
        probability_list = []
        estim_tab = []
        neigh_dict ={}

        #load models rnn
        if mode == 'rnn':
            for midi in range(40,82):
                filename = os.path.join(workspace.model_folder, str(midi) + 'exp_rrn_model.sav')
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
                filename = os.path.join(workspace.model_folder, str(midi) + 'gnb_model.sav')
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
                        sf.write(os.path.join(workspace.workspace_folder,'good',str(cnt_g) + 'good'+str(midi_note.prediction) +'.wav'), instance_data, self.sr, 'PCM_24')
                else:
                    estim_tab.append([midi_note.prediction,7, 
                                        7,onset.prediction, offset])
                    probability_list.append([7])
                    cnt_b += 1
                    sf.write(os.path.join(workspace.workspace_folder,'bad', str(cnt_b) + 'bad'+str(midi_note.prediction) +'.wav'), instance_data, self.sr, 'PCM_24')

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
        self.onsets = [Onset(x) for x in onsets]
        self.midi_notes = [MidiNote(x) for x in midi_notes]
        self.strings = [String(x) for x in strings]
    
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

def compute_confusion_2(confusion_matrix, predicted_tablature, true_tablature, onset_window = 0.025, mode = 'ga'):
    # fix onsets checking before. Keep only the true tablature onsets, and designate wrong onset in case of difference
    false_negative = 0
    for instance in zip(true_tablature.onsets, true_tablature.midi_notes,
                             true_tablature.strings):
        instance = [x.prediction for x in instance]
        t_onset, t_midi, t_string = instance
        p_onset, p_midi, p_string = match_onset(t_onset, predicted_tablature)
        if (p_onset, p_midi, p_string) != (1,1,1):
      #  if abs(t_onset - p_onset)<onset_window: # here add to propagate smallest window in case onset should be taken seriously
            if p_midi != t_midi:
                confusion_matrix[t_string][6] +=1
            elif p_string == 7: # designated the inconclusive
                confusion_matrix[t_string][8] +=1
            elif p_midi == t_midi:
                confusion_matrix[t_string][p_string] +=1
        else:
            confusion_matrix[0][7] += 1
            false_negative += 1
            print('false negatives are ', false_negative)
    return confusion_matrix

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
                confusion_matrix[t_string][7] +=1
            elif p_midi == t_midi:
                confusion_matrix[t_string][p_string] +=1
            
    return confusion_matrix

def compute_confusion_matrixes():
    y_classes = ['E','A','D','G','B','e']
    x_classes = ['E','A','D','G','B','e', 'wrong_pitch','Inconclusive']#,'false_neg_ons'
    correct_ga = 0
    correct_rnn = 0
    confusion_matrix_ga = np.zeros((6,7)) 
    confusion_matrix_rnn = np.zeros((6,8)) 
    jam_list = glob.glob(os.path.join(workspace.annotations_folder, 'single_notes','*solo*.jams'))
  #  jam_list = random.choices(glob.glob(workspace.annotations_folder + '/single_notes/*solo*.jams'), k = 2)
    for index, jam_name in enumerate(jam_list):
        print('we are at {} % '.format(index/len(jam_list)*100))
    #jam_name = workspace.annotations_folder+'/05_BN1-147-Gb_solo.jams'
        x = TrackInstance(jam_name, dataset)
        x.predict_tablature('FromAnnos')
      #  x.get_accuracy_of_prediction('FromCNN')
    #x.rnn_tablature_FromAnnos.tablaturize()
    #x.predicted_tablature_FromAnnos.tablaturize()
        confusion_matrix_ga = compute_confusion(confusion_matrix_ga, x.predicted_tablature_FromAnnos, x.true_tablature)
        confusion_matrix_rnn = compute_confusion(confusion_matrix_rnn, x.rnn_tablature_FromAnnos, x.true_tablature)
        print(confusion_matrix_ga)
    for i in range(6):
        correct_ga += confusion_matrix_ga[i][i]
        correct_rnn += confusion_matrix_rnn[i][i]
    accuracy_ga = correct_ga/np.sum(confusion_matrix_ga)
    accuracy_rnn = correct_rnn/np.sum(confusion_matrix_rnn)
    print(confusion_matrix_rnn)
    print(confusion_matrix_ga)
    title = 'GA Confusion Matrix For +0.02' + dataset + ' Recordings accuracy is ' + str(accuracy_ga)
    plot_confusion_matrix(confusion_matrix_ga, x_classes[:6], y_classes,
                            normalize = True, title = title)
    title = 'Rnn Confusion Matrix For +0.02' + dataset + ' Recordings accuracy is ' + str(accuracy_rnn)
    plot_confusion_matrix(confusion_matrix_rnn, x_classes, y_classes,
                            normalize = True, title = title)

class cnt_help():
    def __init__(self):
        self.cStringWrMidi = 0
        self.cMidiWrString = 0
        self.CStringCMidi = 0
        self.cMidiInc = 0
        self.wrMidiInc = 0
        self.wrStringWrMidi = 0
        self.total = 0

def compare_tabs(AnnosTab, OnsetTab, TrueTab, cnt):
    for instance in zip(AnnosTab.midi_notes, AnnosTab.strings,
                             OnsetTab.midi_notes, OnsetTab.strings, 
                             TrueTab.midi_notes, TrueTab.strings):
        instance = [x.prediction for x in instance]
        a_midi,a_string, o_midi, o_string, t_midi, t_string = instance
        if t_midi == o_midi:
            if o_string == 7:
                cnt.cMidiInc += 1
            elif t_string == o_string:
                cnt.CStringCMidi += 1
            elif t_string != o_string:
                cnt.cMidiWrString += 1
        elif t_midi != o_midi:
            if a_string == 7:
                cnt.wrMidiInc += 1
            elif t_string == a_string: # check from annotations here so that wrong midi doesnt affect
                cnt.cStringWrMidi += 1
            elif t_string != a_string:
                cnt.wrStringWrMidi += 1
        cnt.total += 1
    return cnt



def correlate_string_midi():
    jam_list = glob.glob(os.path.join(workspace.annotations_folder, 'single_notes','*solo*.jams'))
  #  jam_list = random.choices(glob.glob(workspace.annotations_folder + '/single_notes/*solo*.jams'), k = 1)
    cnt = cnt_help()
    for index, jam_name in enumerate(jam_list):
        print('we are at {} % '.format(index/len(jam_list)*100))
        x = TrackInstance(jam_name, dataset)
        x.predict_tablature('OnsetAnnos')
        x.predict_tablature('FromAnnos')
        cnt = compare_tabs(x.rnn_tablature_FromAnnos, x.rnn_tablature_FromCNN, x.true_tablature, cnt)
        print('wrong midi wrong string is {}percent of wrong midi correct string is {} correct midi wrong string {} correct midi correct string {} correct midi inconclusive {} wrong midi inconclusive {} and total number of notes{}'.format(cnt.wrStringWrMidi* 100/cnt.total, cnt.cStringWrMidi* 100/cnt.total , cnt.cMidiWrString * 100/cnt.total, cnt.CStringCMidi* 100/cnt.total,cnt.cMidiInc*100/cnt.total, cnt.wrMidiInc*100/cnt.total, cnt.total))
    

def test():                                           
    jam_name = os.path.join(workspace.annotations_folder,'05_BN1-147-Gb_solo.jams')
    x = TrackInstance(jam_name, dataset)
    x.predict_tablature('FromAnnos')
  #  x.rnn_tablature_FromAnnos.tablaturize()
    x.predicted_tablature_FromAnnos.tablaturize()

def conf_pitch(predicted_tablature, true_tablature, conf_pitch_m):

    for instance in zip(predicted_tablature.onsets,
                        predicted_tablature.midi_notes,
                            predicted_tablature.strings,
                            true_tablature.onsets, true_tablature.midi_notes,
                             true_tablature.strings):
        instance = [x.prediction for x in instance]
        p_onset, p_midi, p_string, t_onset, t_midi, t_string = instance
        #if p_midi>39 and p_midi<73:
        if p_midi == t_midi:
            conf_pitch_m[0][0] += 1
        elif abs(p_midi - t_midi) == 1:
            conf_pitch_m[0][1] += 1
        elif (p_midi - t_midi)%12 == 0:
            conf_pitch_m[0][2] += 1
        else:
            conf_pitch_m[0][3] += 1
    print(conf_pitch_m)
    return conf_pitch_m

def check_pitch():
    x_classes = ['correct', 'wrong by 1', 'wrong by octave', 'wrong by more']
    conf_pitch_m = np.zeros((1,4))
    jam_list = glob.glob(os.path.join(workspace.annotations_folder, 'single_notes','*solo*.jams'))
   # jam_list = random.choices(glob.glob(workspace.annotations_folder + '/single_notes/*solo*.jams'), k = 1)
    cnt = cnt_help()
    for index, jam_name in enumerate(jam_list):
        print('we are at {} % '.format(index/len(jam_list)*100))
        x = TrackInstance(jam_name, dataset)
        x.predict_tablature('OnsetAnnos')
        conf_pitch_m = conf_pitch(x.predicted_tablature_OnsetAnnos, x.true_tablature, conf_pitch_m)
    
    accuracy_pitch = np.trace(conf_pitch_m)
    print(conf_pitch_m)
    title = 'Pitch confusion Matrix For ' + dataset + ' Recordings accuracy is ' + str(accuracy_pitch)+'No Stop'
    plot_confusion_matrix(conf_pitch_m, x_classes, x_classes[:1],
                            normalize = True, title = title)



if __name__ == "__main__":
   # correlate_string_midi()
   # check_pitch()
    compute_confusion_matrixes()
 # test()
   # jam_name = os.path.join(workspace.annotations_folder,'05_Rock1-130-A_solo.jams') #05_BN1-147-Gb_solo
    
    '''x = TrackInstance(os.path.join('C:\\','Users','stefa','Documents','guit_workspace',
                                    'IDMT-SMT-GUITAR_V2', 'dataset2','audio',
                                        'AR_A_fret_0-20.wav'), dataset, AnnosMode = False)
    x.predict_tablature('FromCNN')
    x.predicted_tablature_FromCNN.tablaturize()
    x.predicted_tablature_FromCNN.output_to_midi()'''