import pandas as pd

class Optimizer(object):
    def __init__(self, params, **kwargs):
        self.params = params
        # to keep the cmpds that already have their costs.
        self.retained_population = pd.DataFrame()

    def optimize(self):
        """
        Optimize the molecule cost in the latent space and returns optimized latent variables
        Args:
            latent_cost_df (dataframe): molecules presented by latent variables and calculated cost of the molecules.
            latent_cost_df must have columns ['latent', 'cost']
        Raises:
            NotImplementedError: The method is implemented by subclasses
        """
        raise NotImplementedError

"""
class GeneticOptimizer(Optimizer):
    def __init__(self, params):
        self.retained_population = pd.DataFrame()
        self.selection_type = params.selection_type.lower()
        self.tourn_size = params.tourn_size
        self.mate_prob = params.mate_prob
        self.max_population = params.max_population
        self.optima_type = params.optima_type
        self.elite_perc = params.elite_perc

    def optimize(self, population):
        self.population = pd.concat([population, self.retained_population])
        self.population.reset_index(drop=True, inplace=True)
        self.population_size = len(self.population)
        self.chromosome_length = len(self.population["chromosome"].iloc[0]) #STB: aligning with GA language in generalizing code

        print(f"Combined population size: {self.population_size}, new population size: {len(population)}, retained_df size: {len(self.retained_population)}, target population size: {self.max_population}")
        
        self.select()
        return self.population

    def set_retained_population(self, unchanged_individuals):
        self.retained_population = unchanged_individuals

    def tournament_selection(self, selection_pool):
        selection_pool[["fitness", "compound_id"]] = selection_pool[["fitness", "compound_id"]].apply(pd.to_numeric)
        
        if self.optima_type.lower() == "minima":
            return selection_pool.fitness.idxmin()
        elif self.optima_type.lower() == "maxima":
            return selection_pool.fitness.idxmax()
        else:
            raise ValueError(f"Unknown optima type {self.optima_type.lower()}")

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
"""

"""
class ParticleSwarmOptimizer(Optimizer):
    def __init__(self):
        pass
    def optimize(self):
        pass
"""

"""
def test_selection():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', help="Config file location *.yml", action='append', required=True)
    args = parser.parse_args()

    for conf_fname in args.config:
        with open(conf_fname, 'r') as f:
            parser.set_defaults(**yaml.load(f, Loader=Loader))

    params = parser.parse_args()

    population = pd.read_csv("/mnt/projects/ATOM/blackst/FNLGMD/workspace/development/debug_population.csv")
    
    optimizer = create_optimizer(params)
    population = optimizer.optimize(population)

    population.to_csv("/mnt/projects/ATOM/blackst/FNLGMD/workspace/development/optimized.csv", index=False)

if __name__ == "__main__":
    test_selection()
"""