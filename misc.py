import os
from initialize_workspace import *
import pandas as pd
import random
from helper import determine_combinations
import math
import numpy as np
import matplotlib.pyplot as plt
import itertools

workspace = initialize_workspace(os.path.join('C:\\','Users\stefa\Documents\guit_workspace'))

def get_probs(prob, classes):
    g = 0
    res = []
    for k in range(0,6):
        if k in classes:
            res.append(prob[g])
            g+=1
        else:
            res.append(0)
    return res


def compute_fret(string,midi):
    string_dict = {0 : 40, 1 :  45, 2 : 50 ,
                3 : 55 , 4 :  59, 5 :64}
    if string == 7:
        return 7
    return int(midi-string_dict[string])

def read_correlate_matrix():
    r = []
    dfd = my_read_csv('fullcorelation_matrix_all_tracks_1000_perms.csv')
    if dfd is not None:
        res = dfd.to_numpy()    
        #    print(res)
    for row in res[:-1,1:-1]:
        r.append(1/sum(row))
    return r

def my_read_csv(name):
    if os.path.isfile(os.path.join(workspace.result_folder, 'call', name)):
        df = pd.read_csv(os.path.join(workspace.result_folder, 'call', name))
        return df
    else:
        return None


#===========necessary for genetic algorithm===========================

def init_tab(estim_tab):
    res_tab = []
    for index, instance in enumerate(estim_tab):
        if instance[1] == 7:
            t = list(random.choice(determine_combinations(instance[0])))[0:3]
            t.extend([instance[3],instance[4]])
            res_tab.append(t)
        else:
            res_tab.append(instance)
    mutate_func(res_tab, mut_rate=0.3)
    return res_tab

def change_spot(place):
    midi, string, fret, start, end = place
    t = list(random.choice(determine_combinations(midi)))[0:3]
    t.extend([start,end])
    #place[0:2] = t[1:3]
    return t

def permutate_tablature(index_range, init_tab):
    fin_tab = []
    for index in index_range:
        temp = []
    #    print(init_tab[index])
        temp = change_spot(init_tab[index])
        fin_tab.append(temp)
    return fin_tab

def mutate_func(individual,mut_rate=0.1):
    fin_tab = []
    index_range = len(individual)
    for index in range(index_range):
        if random.random() < mut_rate:
            individual[index] = (change_spot(individual[index]))
    #print(fin_tab)
    return individual

def get_tru(tru_tab):
    return tru_tab

#===============fitness evaluation====================
def faster_weight(tablature, coeff):
    tot_mean = 10.11 # without coefficients
    tot_std = 1.02
   # tot_mean = 3.29 with coefficients on computation
   # tot_std = 0.33
    '''mean_vals = [ 0.87697473, 1.18229851, 1.353474, 0.9201629, -0.03543685]
    std_vals = [0.10412535, 0.20857569, 0.40880753, 0.54117571, 0.23679024] mean and std of tru tabs only'''
    mean_vals = [ 0.947081603,  3.21093206,  4.77504591,  1.22841735, -0.0553669747]
    std_vals = [0.0397906105, 0.344142254, 0.500706338, 0.282981956, 0.0527832055]
    #a,b,c,d,e = 1,1,1,1,1
    a,b,c,d,e = coeff
    total_weight = 0
    start, end = 0, 0
    weight_open = 0
    weight_string = 0
    fret_weight = 0
    depress_weight = 0
    previous = tablature[0]
    for index, current in enumerate(tablature):
        #   open string
        if current[2]==0:
            weight_open-=1
# string factor
        if index>0:
            weight_string += abs(current[1]-previous[1])
#fret factor
            tim_f = 1/(((current[3]-previous[3])*10)+0.2)
            fret_weight += (abs(current[2]-previous[2]))#*tim_f)**3#*(1/(((current[2]*10**3-previous[2]*10**3))**2))# assuming 50ms per note,(fastest way) 20*50 = 1 so weight is doubled for fastest playable note
            norm = 1
#depress factor
            if previous[0:2]==current[0:2]:
                pass
            else:
                depress_weight+=1
            previous = current
# averge fret
        back_range = list(range(index-2,index+1))
        back_range.reverse()
        front_range = list(range(index,index+3))

        for i, k in enumerate(back_range):
            if k<=0:
                break
            elif i == (len(back_range)-1):
                start = k
                break
            else:
                time_step = tablature[back_range[i]][3]-tablature[back_range[i+1]][3]
                if time_step>1:
                    start = k
                    break
    
                start = k
        for i, k in enumerate(front_range):
            if k>(len(tablature)-1):
                break
            elif k == (len(tablature)-1):
                end = k
                break
            elif i == (len(front_range)-1):
                end = k
                break
            else:
                time_step = tablature[front_range[i+1]][3]-tablature[front_range[i]][3]
                if time_step>1:
                    end = k
                    break
                end = k
        if end != start:
            _, string_l, fret_l, _,_ = zip(*tablature[start:end])
            avg_fret = sum(fret_l)/len(fret_l)
            avg_string = sum(string_l)/len(fret_l)
            avg_weight = math.sqrt((avg_fret-current[2])**2+(avg_string-current[1])**2)
            total_weight += avg_weight
    
    open_f = weight_open/len(tablature)
    avg = total_weight/len(tablature)
    string_f = weight_string/len(tablature)
    depress = depress_weight/len(tablature)
    fret_f = fret_weight/(norm*len(tablature))
    '''
    open_f = (weight_open/len(tablature) - mean_vals[4]) / std_vals[4]
    avg = (total_weight/len(tablature) - mean_vals[1]) / std_vals[1]
    string_f = (weight_string/len(tablature) - mean_vals[3]) / std_vals[3]
    depress = (depress_weight/len(tablature) - mean_vals[0]) / std_vals[0]
    fret_f = (fret_weight/(norm*len(tablature)) - mean_vals[2]) / std_vals[2]
    '''
   # print(depress,avg,fret_f,string_f,open_f)
    #return a*depress+b*avg+c*fret_f+d*string_f+e*open_f
    return (a*depress+b*avg+c*fret_f+d*string_f+e*open_f)# - tot_mean) / tot_std

def evaluate(individual, estim_tab, probability_list, coeff): # fix so it permuatates through other

    return get_full_weight(individual, estim_tab, probability_list, coeff), 0

def get_full_weight(tab_check, estim_tab, probability_list, coeff=(1,1,1,1,1)):
    est_m = 0.7597065171781754
    est_std = 0.09836368204855489
    c_est = 2*sum(coeff)/(len(coeff))
    est = get_weight_of_tab(tab_check, estim_tab, probability_list)
    fret_we = faster_weight(tab_check, coeff)
    return  fret_we - c_est*est#*(est-est_m)/est_std

def get_weight_of_tab(tru_tab, estim_tab, mode = 'plain', probability_list = None):
    if probability_list is None:
        probability_list = list([[1,1,1,1,1,1]]*len(estim_tab))
    cnt = 0
    wr = 0
    for index, instance in enumerate(estim_tab):
        if instance[1]==7:
            wr+=1
            #pass
        elif tru_tab[index][1]==instance[1]:
            if mode =='n':
                cnt+= 1 #probability_list[index][instance[1]] # maybe multiply with a factor, to induce more difference depending on probability
            else:
                cnt+= 1#probability_list[index][instance[1]]#**2
        else:
            if mode == 'minus':
                cnt-=1
    return cnt/(len(estim_tab)-wr)

#============= other============
def determine_combinations(midi_f_cand): # returns all posible ways to play a candidate fundamental
    ret = []
    fret_range = [range(40,56),range(45,61),range(50,66),range(55,71),range(59,75),range(64,82)]
    fret_range = [range(40,58),range(45,63),range(50,68),range(55,73),range(59,77),range(64,82)]
    #fret_range = [range(40,53),range(45,58),range(50,63),range(55,68),range(59,72),range(64,77)]
    for string, x in enumerate(fret_range):
        if midi_f_cand in list(x):
            ret.append((midi_f_cand, string, compute_fret(string, midi_f_cand)))
    return ret

    #=== plotting and statistics

def plot_confusion_matrix(cm, x_classes,y_classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.clf()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        np.nan_to_num(cm,False)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks_y = np.arange(len(y_classes))
    tick_marks_x = np.arange(len(x_classes))
    plt.xticks(tick_marks_x,x_classes , rotation=45)
    plt.yticks(tick_marks_y, y_classes)

    fmt = '.2f' if normalize else '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(workspace.result_folder, title.replace(" ", "") +'.png'))
    return plt