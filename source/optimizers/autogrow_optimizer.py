import pandas as pd
import copy
from optimizers.base_optimizer import Optimizer

class AutoGrowOptimizer(Optimizer):
    def __init__(self, params):
        self.tourn_size = params.tourn_size

        self.optima_type = params.optima_type.lower()
        if self.optima_type not in ['minima', 'maxima']:
            raise ValueError(f"Unknown optima type {self.optima_type}. Available options are: {['minima', 'maxima']}")
        
        self.selection_type = params.selection_type.lower()
        if self.selection_type not in ['tournament', 'roulette']:
            raise ValueError(f"Unknown optima type {self.selection_type}. Available options are: {['tournament', 'roulette']}")
        
    def select_elite_pop(self, population, elite_size):

        if self.optima_type == "minima":
            sort_order = True
        elif self.optima_type == "maxima":
            sort_order = False

        selected_population = population.sort_values(by=['fitness'], ascending = sort_order)
        selected_population = selected_population.head(elite_size)

        return selected_population

    def select_non_elite(self, population, size):

        self.population = copy.deepcopy(population)
        selected_population = pd.DataFrame(columns=['compound_id', 'smiles', 'generation', 'chromosome', 'fitness'])

        while len(selected_population) < size:
            #Setup the pool of individuals for the tournament selection to be a random sampling of the population
            #Without replacement means that the same individual will not appear in the sampling more than once
            if self.tourn_size > len(self.population):
                selection_pool = self.population.sample(len(self.population), replace=False) #TODO: Can create an alternative WITH replacement
            else:
                selection_pool = self.population.sample(self.tourn_size, replace=False) #TODO: Can create an alternative WITH replacement

            if self.selection_type == "tournament":
                selected_individual = self.tournament_selection(selection_pool)
            elif self.selection_type == "roulette":
                selected_individual = self.roulette_selection(selection_pool)

            #Now we can add the selected individual to the selected_population and remove it from the self.population
            # so that we do not have repeated individuals in our selected population. This can lead to false convergence

            selected_population.loc[len(selected_population.index)] = self.population.loc[selected_individual]
            #TODO: Do we need to allow repeated individuals as the population converges? 
            self.population.drop([selected_individual], inplace=True) 
        
        #Reset the self.population to contain the selected individuals that will be used for creation of next generation
        self.population = selected_population.reset_index(drop=True)
    
        return selected_population
    
    def tournament_selection(self, selection_pool):
        selection_pool[["fitness", "compound_id"]] = selection_pool[["fitness", "compound_id"]].apply(pd.to_numeric)
        
        if self.optima_type == "minima":
            return selection_pool.fitness.idxmin()
        elif self.optima_type == "maxima":
            return selection_pool.fitness.idxmax()
    
    def roulette_selection(self):
        pass
