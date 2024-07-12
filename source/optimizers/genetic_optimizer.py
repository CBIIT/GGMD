import copy
import pandas as pd
import numpy as np
from optimizers.base_optimizer import Optimizer

class GeneticOptimizer(Optimizer):
    def __init__(self, params):
        self.tourn_size = params.tourn_size

        self.selection_type = params.selection_type.lower()
        if self.selection_type not in ['tournament', 'roulette']:
            raise ValueError(f"Unknown optima type {self.selection_type}. Available options are: {['tournament', 'roulette']}")

        self.optima_type = params.optima_type.lower()
        if self.optima_type not in ['minima', 'maxima']:
            raise ValueError(f"Unknown optima type {self.optima_type}. Available options are: {['minima', 'maxima']}")
        
        #self.sample_with_replacement = params.sample_with_replacement

    def tournament_selection(self, selection_pool):
        selection_pool[["fitness"]] = selection_pool[["fitness"]].apply(pd.to_numeric)
        
        if self.optima_type == "minima":
            return selection_pool.fitness.idxmin()
        elif self.optima_type == "maxima":
            return selection_pool.fitness.idxmax()
    
    def roulette_selection(self, selection_pool):
        #IN WORK
        max_val = selection_pool['fitness'].max()
        
        fitnes_values = selection_pool['fitness'].tolist()
        weights = [f/max_val for f in fitnes_values]
        #IN WORK
        return np.random.choice()

    def select_non_elite(self, population, size):
        self.population = copy.deepcopy(population)
        selected_population = pd.DataFrame(columns=['compound_id', 'smiles', 'generation', 'chromosome', 'fitness'])
        
        current_index = 0

        while len(selected_population) < size:
            if len(self.population) == 0:
                self.population = copy.deepcopy(population)
            #Setup the pool of individuals for the tournament selection to be a random sampling of the population
            #Without replacement means that the same individual will not appear in the sampling more than once
            if self.tourn_size > len(self.population):
                selection_pool = self.population.sample(len(self.population), replace=False) #TODO: Can create an alternative WITH replacement
            else:
                selection_pool = self.population.sample(self.tourn_size, replace=False) #TODO: Can create an alternative WITH replacement
            
            if self.selection_type == "tournament":
                #print("self.population.shape ", self.population.shape, " selected_population.shape ", selected_population.shape)
                selected_individual = self.tournament_selection(selection_pool)
            elif self.selection_type == "roulette":
                selected_individual = self.roulette_selection(selection_pool)

            """ Testing, code still in work
            #Sampling without replacement means that more fit individuals will be more likely to be selected to be parents 
            # more than once. After they are selected, they are not removed from the pool.
            if self.sample_with_replacement == False:
                #If we aren't removing selected parents, we need to ensure that parent1 and parent2 aren't the same smiles string. 
                # This is only relevant on odd index values as parents are ordered for crossover. For example, 
                # parent1 will be index 0, parent2 will be index 1. So, if index 1 and 0 in the selected population have the same smiles string,
                # crossover will be pointless. 
                if current_index % 2 != 0: 
                    
                    if selected_population.loc[current_index - 1, 'smiles'] == selection_pool.loc[selected_individual, 'smiles']:
                        continue

            #Now we can add the selected individual to the selected_population
            #selected_population.loc[len(selected_population.index)] = selection_pool.loc[selected_individual]
            selected_population.loc[current_index] = selection_pool.loc[selected_individual]
            current_index += 1

            #Sampling with replacement means that once an individual is selected to be a parent, it will be removed from the pool 
            # of candidates for parents. Each individual will only be selected to be a parent once unless selection is done more 
            # than once. 
            if self.sample_with_replacement == True:
                self.population.drop([selected_individual], inplace=True)
            """

            #Now we can add the selected individual to the selected_population
            selected_population.loc[len(selected_population.index)] = selection_pool.loc[selected_individual]

            self.population.drop([selected_individual], inplace=True)
        
        #Reset the self.population to contain the selected individuals that will be used for creation of next generation
        self.population = selected_population.reset_index(drop=True)
        #print("Population_size after selection ", len(self.population))

        return selected_population

    def select_elite_pop(self, population, size):
        population = copy.deepcopy(population)
        
        if self.optima_type == "minima":
            sort_order = True
        elif self.optima_type == "maxima":
            sort_order = False

        selected_population = population.sort_values(by=['fitness'], ascending=sort_order)
        selected_population = selected_population.head(size)

        return selected_population