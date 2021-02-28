# Fresh Guit Proc

# Guitar String Detection Using Machine Learning and Genetic Algorithms

## libraries:
madmom(https://github.com/CPJKU/madmom), crepe (https://github.com/marl/crepe), librosa, pandas, DEEP (https://github.com/DEAP/deap), multiprocessing (https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing), pickle, sklearn, scipy
## Set up workspace:
Download dataset from https://zenodo.org/record/3371780. Contains Annotations, and 3 datasets corresponding to same track, with a different recording method. 6chanel-debleeded, Summed chanels, and microphone recordings.  
Choose a workspace folder* ~/PathTo/MyWorkspace*. Create directories: 
- *~/PathTo/MyWorkspace/annos* ->extract annotations here 
- *~/PathTo/MyWorkspace/mic* ->mic recordings 
- *~/PathTo/MyWorkspace/hex_cln* -> 6 chanel-debleeded 
- *~/PathTo/MyWorkspace/audio_mono_pickup* -> summed chanels  # mix is a better name
++ models
++ results

## Scripts:
### track_class.py:
**Change line!!!** (currently 20)->  workspace = initialize_workspace('/media/estfa/10dcab7d-9e9c-4891-b237-8e2da4d5a8f2/data_2')
to correspond to* ~/PathTo/MyWorkspace*. 
*track_class.py* sums up, all methods to predict a tablature given the onsets, pitch, and recording. To do that:  

`jam_name = workspace.annotations_folder+'/jam_name.jams`  
`x = TrackInstance(jam_name, dataset)`  
`x.predict_tablature('FromAnnos')   # can be exchanged FromCNN to predict onsets and pitch using the madmom and crepe libraries`    

dataset of choice is also hardcoded as* 'mic'* to change that, change (currently line 19) 'dataset = 'mic' ' to* 'audio_mono_pickup'* or* 'hex_cln'*

### helper.py
Contains key components of the script.Crucial Among them,  is the note_instance class with the methods, **compute_partials** and **compute_beta**. Example use:  

`x = note_instance(name = self.track_name, data = instance_data, midi_note = midi_note.prediction)`  
`x.compute_partials(no_of_partials,diviate=x.fundamental_measured/2)`  
`beta = compute_beta(y=np.array(x.differences),track=x)`  

(keyword arguement *diviate* is the length of the window used to detect partials, as proposed by Barbancho et al, we set it to half of the fundamental)
(compute_beta function's primary input, *(y=np.array(x.differences)* is the differences of the measured partials from the expected ones and the corresponding partial's number. 

### genetic.py:
Holds the genetic function that takes as input an estimated guitar tab, and uses guitar playability constraints, to improve results. 

 `genetic(estim_tab, probability_list, coeff)` 
 
*estim_tab* corresponds to so far estimated tablature, *probability_list* to confidence of the so far predicted values (currently ignored), and *coeff* to coeeficients paired to each playability constraint. (string distance, fret distance, etc)




run track_class.get_features_and_targets() to create npz files

Onset Detection: python music_test.py --epochs 100 --modality Audio -train --project_dir ~/fresh-guit-proc/ --feats melspec --ksize 5 --instr guitar --fs 44100 --hop 441 --w_size 1764 -rescaled