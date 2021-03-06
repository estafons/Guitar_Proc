import random

def get_midi(string, fret):
    ret = []
    fret_range = [range(40,56),range(45,61),range(50,66),range(55,71),range(59,75),range(64,82)]
    for index, midi in enumerate(fret_range[string]):
        if index == fret:
            break
    return midi

def init_dict_of_dicts():
    dict_of_dicts ={}
    for midi in range(40,85):
        dict_of_dicts[midi] = {0:[],1:[],2:[],3:[],4:[],5:[]}
    return dict_of_dicts

def beta_get(init_beta, fret):
    theory_b = init_beta*2**(fret/6)# or 12?
    sigma = theory_b/5
    mu = theory_b
    return random.gauss(mu, sigma)

def simulate_betas():
    beta_dict={0 : 0.00011735, 1: 8.617*10**(-5), 2 : 4.6731*10**(-5) , 
                3 : 2.5995*10**(-5) , 4 : 6.438*10**(-5) , 5 : 2.177*10**(-5)}
    dict_of_dicts = init_dict_of_dicts()
    for string in range(0,6):
        for fret in range(0,18):
            midi = get_midi(string, fret)
            for i in range(50):
                beta = beta_get(beta_dict[string], fret)
                dict_of_dicts[midi][string].append(beta)
    return dict_of_dicts

