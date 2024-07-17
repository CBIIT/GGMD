import copy
import pandas as pd
import numpy as np
from optimizers.base_optimizer import Optimizer

class GeneticOptimizer(Optimizer):
    def __init__(self, params):
        self.selection_pool_size = params.selection_pool_size

        self.selection_type = params.selection_type.lower()
        if self.selection_type not in ['tournament', 'roulette']:
            raise ValueError(f"Unknown optima type {self.selection_type}. Available options are: {['tournament', 'roulette']}")

        self.optima_type = params.optima_type.lower()
        if self.optima_type not in ['minima', 'maxima']:
            raise ValueError(f"Unknown optima type {self.optima_type}. Available options are: {['minima', 'maxima']}")
        
        self.sample_with_replacement = params.sample_with_replacement
        self.max_clones = params.max_clones
        

    def tournament_selection(self, selection_pool):
        selection_pool[["fitness"]] = selection_pool[["fitness"]].apply(pd.to_numeric)
        
        if self.optima_type == "minima":
            return selection_pool.fitness.idxmin()
        elif self.optima_type == "maxima":
            return selection_pool.fitness.idxmax()
    
    def roulette_selection(self, selection_pool):
        #IN WORK
        fitness_scores = selection_pool['fitness'].tolist()
        min_fit = min(fitness_scores)
        shifted_scores = [score - min_fit for score in fitness_scores]

        total = sum(shifted_scores)
        normalized_scores = [score / total for score in shifted_scores]

        #IN WORK
        return selection_pool.sample(1, weights=normalized_scores).index[0]

    def select_non_elite(self, population, size):
        population_pool = copy.deepcopy(population)
        selected_population = pd.DataFrame(columns=['compound_id', 'smiles', 'chromosome', 'fitness'])
        
        current_index = 0

        while len(selected_population) < size:
            if len(population_pool) == 0:
                population_pool = copy.deepcopy(population)
                
            #Setup the pool of individuals for the tournament selection to be a random sampling of the population
            #Without replacement means that the same individual will not appear in the sampling more than once
            if self.selection_pool_size > len(population_pool):
                selection_pool = population_pool.sample(len(population_pool), replace=False) #TODO: Can create an alternative WITH replacement
            else:
                selection_pool = population_pool.sample(self.selection_pool_size, replace=False) #TODO: Can create an alternative WITH replacement
            
            if self.selection_type == "tournament":
                selected_individual = self.tournament_selection(selection_pool)
            elif self.selection_type == "roulette":
                selected_individual = self.roulette_selection(selection_pool)

            selected_ind_smiles = selection_pool.loc[selected_individual, 'smiles']
            
            if current_index % 2 != 0: 
                last_selected_smiles = selected_population.loc[current_index - 1, 'smiles'] 

                if last_selected_smiles in self.clone_tracker:
                    if selected_ind_smiles in self.clone_tracker[last_selected_smiles]:
                        if self.clone_tracker[last_selected_smiles][selected_ind_smiles] >= self.max_clones:
                            continue
                        else:
                            self.clone_tracker[last_selected_smiles][selected_ind_smiles] += 1

                    else:
                        self.clone_tracker[last_selected_smiles][selected_ind_smiles] = 1

                elif selected_ind_smiles in self.clone_tracker:
                    if last_selected_smiles in self.clone_tracker[selected_ind_smiles]:
                        if self.clone_tracker[selected_ind_smiles][last_selected_smiles] >= self.max_clones:
                            continue

                        else:
                            self.clone_tracker[last_selected_smiles][selected_ind_smiles] += 1
                    else:
                        self.clone_tracker[selected_ind_smiles][last_selected_smiles] = 1

                else:
                    self.clone_tracker[last_selected_smiles] = {selected_ind_smiles: 1}
            
            #Now we can add the selected individual to the selected_population
            selected_population = pd.concat([selected_population, selection_pool.loc[selected_individual].to_frame().T], ignore_index=True)
    
            current_index += 1

            #Sampling with replacement means that once an individual is selected to be a parent, it will be removed from the pool 
            # of candidates for parents. Each individual will only be selected to be a parent once unless selection is done more 
            # than once. 
            if self.sample_with_replacement == True:
                population_pool.drop([selected_individual], inplace=True)
        
        #Reset the population_pool to contain the selected individuals that will be used for creation of next generation
        population_pool = selected_population.reset_index(drop=True)

        return selected_population

    def select_elite_pop(self, population, size):
        population = copy.deepcopy(population)
        self.clone_tracker = {}
        
        if self.optima_type == "minima":
            sort_order = True
        elif self.optima_type == "maxima":
            sort_order = False

        selected_population = population.sort_values(by=['fitness'], ascending=sort_order)
        selected_population = selected_population.head(size)

        return selected_population