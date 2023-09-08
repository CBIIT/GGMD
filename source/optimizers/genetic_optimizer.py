import pandas as pd
from optimizers.base_optimizer import Optimizer

class GeneticOptimizer(Optimizer):
    def __init__(self, params):
        self.retained_population = pd.DataFrame()
        self.selection_type = params.selection_type.lower()
        self.tourn_size = params.tourn_size
        self.mate_prob = params.mate_prob
        self.max_population = params.max_population
        self.optima_type = params.optima_type.lower()
        self.elite_perc = params.elite_perc

    def optimize(self, population):
        self.population = pd.concat([population, self.retained_population])
        self.population.reset_index(drop=True, inplace=True)
        self.population_size = len(self.population)
        #self.chromosome_length = len(self.population["chromosome"].iloc[0]) #STB: aligning with GA language in generalizing code

        print(f"Combined population size: {self.population_size}, new population size: {len(population)}, retained_df size: {len(self.retained_population)}, target population size: {self.max_population}")
        
        self.select()
        return self.population

    def set_retained_population(self, unchanged_individuals):
        self.retained_population = unchanged_individuals

    def tournament_selection(self, selection_pool):
        selection_pool[["fitness", "compound_id"]] = selection_pool[["fitness", "compound_id"]].apply(pd.to_numeric)
        
        if self.optima_type == "minima":
            return selection_pool.fitness.idxmin()
        elif self.optima_type == "maxima":
            return selection_pool.fitness.idxmax()
        else:
            raise ValueError(f"Unknown optima type {self.optima_type}")

    def select(self):
        num_survived = int(self.mate_prob * self.population_size) #int((1 - 0.3) * 50) = 35
        print("num_survived", num_survived)
        selected_population = pd.DataFrame(columns=['compound_id', 'smiles', 'chromosome', 'fitness'])

        while len(selected_population) < num_survived:
            #Setup the pool of individuals for the tournament selection to be a random sampling of the population
            #Without replacement means that the same individual will not appear in the sampling more than once
            if self.tourn_size > len(self.population):
                selection_pool = self.population.sample(len(self.population), replace=False) #TODO: Can create an alternative WITH replacement
            else:
                selection_pool = self.population.sample(self.tourn_size, replace=False) #TODO: Can create an alternative WITH replacement
            
            selected_individual = self.tournament_selection(selection_pool)

            #Now we can add the selected individual to the selected_population and remove it from the self.population
            # so that we do not have repeated individuals in our selected population. This can lead to false convergence
            
            selected_population.loc[len(selected_population.index)] = selection_pool.loc[selected_individual]
            #TODO: Do we need to allow repeated individuals as the population converges? 
            self.population.drop([selected_individual], inplace=True) 
        
        #Reset the self.population to contain the selected individuals that will be used for creation of next generation
        self.population = selected_population.reset_index(drop=True)
        print("Population_size after selection ", len(self.population))