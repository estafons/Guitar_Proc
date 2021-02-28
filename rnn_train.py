from misc import determine_combinations
from sklearn.neighbors import RadiusNeighborsClassifier
import pickle
from simulate_training import simulate_betas
import numpy as np
from initialize_workspace import *

dataset = 'mic'
track_extension = '_hex_cln'
mode_is ='full'
workspace = initialize_workspace(os.path.join('C:\\','Users\stefa\Documents\guit_workspace'))

def get_radius(midi):
    beta_list = []
    beta_dict={0 : 0.00011735, 1: 8.617*10**(-5), 2 : 4.6731*10**(-5) , 
                3 : 2.5995*10**(-5) , 4 : 6.438*10**(-5) , 5 : 2.177*10**(-5)}
    combs = determine_combinations(midi)
    for comb in combs:
        _, string, fret = comb
        beta_list.append(beta_dict[int(string)]*2**(fret/6))
    return get_smallest_dif(beta_list)

def get_smallest_dif(beta_list):
    previous = beta_list[0]
    min_so_far = 1
    for index, beta in enumerate(beta_list[1:]):
        current = beta
        if abs(current-previous)<min_so_far:
            min_so_far = abs(current-previous)
    if min_so_far == 1:
        min_so_far = beta_list[0]/2
    return min_so_far/4

def train_models(dict_of_dicts):
    for midi_key, midi_dict in dict_of_dicts.items():
        X = []
        y = []
        for x in midi_dict:
            if midi_dict[x] == []:
                pass
            else:
                for t in midi_dict[x]:
                    X.append(t)
                    y.append(x) # n times
        if X != []:
            rad = get_radius(midi_key)

            neigh = RadiusNeighborsClassifier(radius = rad,weights='distance', outlier_label=[7])

            X = np.array(X)
            y = np.array(y)
            neigh.fit(X.reshape(-1, 1), y)
            filename = os.path.join(workspace.model_folder, str(midi_key) + 'rrn_model.sav')
            pickle.dump(neigh, open(filename, 'wb'))
        else:
            pass
if __name__ == "__main__":
    dict_of_dicts = simulate_betas()
    train_models(dict_of_dicts)