#from misc import *
from deap import  base, creator, tools
from misc import init_tab, mutate_func, evaluate, faster_weight
import multiprocessing
import random

def genetic(estim_tab, probability_list, coeff):
    NGEN = 300
    CXPB = 0.5
    MUTPB = 0.2
    mut = 0.1
    max_so_far = 0
    pop = []
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # create toolbox
    toolbox = base.Toolbox()
   # toolbox.register("get_tru", get_tru, tru_tab)
   # toolbox.register("cor_individual", tools.initIterate, creator.Individual,
    #                toolbox.get_tru)

    toolbox.register("init_tab", init_tab, estim_tab)
  #  toolbox.register("perm_tab", permutate_tablature, range(0,len(initial_tab)), initial_tab)
    toolbox.register("individual", tools.initIterate, creator.Individual,
                    toolbox.init_tab)
    #toolbox.register("individual", tools.initIterate, creator.Individual,
    #                toolbox.perm_tab)
    toolbox.register("evaluate", evaluate, estim_tab = estim_tab, probability_list = probability_list, coeff = coeff)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, 40000)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate_func)
    toolbox.register("select", tools.selTournament, k =3000, tournsize = 5)#, #tournsize=3)
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)#
    #create initial population
    pop = toolbox.population()
    #pop.append(toolbox.cor_individual())
    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    for g in range(NGEN):

        offspring = toolbox.select(pop)
        offspring = list(toolbox.map(toolbox.clone, offspring))

        # Apply crossover on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation on the offspring
        for mutant in offspring:
            if random.random() < MUTPB:

                toolbox.mutate(mutant, mut_rate = mut)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        offspring.extend(tools.selBest(pop, 100))
        selected = tools.selBest(offspring, 3000)
        [res] = tools.selBest(selected, 1)
        if g==0:
            print(faster_weight(res, coeff))
        pop[:] = selected
        if toolbox.evaluate(pop[0])==toolbox.evaluate(pop[500]) or g == NGEN-1:
            print(g)
            print(faster_weight(res, coeff))
            break

    [res] = tools.selBest(pop, 1)
    cnt = 0
    print(list(toolbox.map(toolbox.evaluate, pop[:5])))

    return res, g