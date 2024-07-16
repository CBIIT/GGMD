import numpy as np
import pandas as pd
from copy import deepcopy
from optimizers.optimizer_factory import create_optimizer
from generative_networks.base_generative_model import GenerativeModel
from generative_networks.generative_models_helpers.AutoGrow.mutation.execute_mutations import Mutator
from generative_networks.generative_models_helpers.AutoGrow.crossover.execute_crossover import CrossoverOp

class AutoGrow(GenerativeModel):
    def __init__(self, params):
        self.max_population = params.max_population
        self.num_crossovers = params.num_crossovers
        self.num_mutations = params.num_mutations
        self.num_elite = params.num_elite

        self.max_clones = params.max_clones
        
        self.generation_number = 0
        self.optimizer = create_optimizer(params)
        self.mutator = Mutator(params)
        self.CrossoverOp = CrossoverOp(params)

    def mutate(self):
        print("Begin Mutation")
        mutated_smiles_df = pd.DataFrame({"smiles": [], "parent1_id": [], "reaction_id": [], "zinc_id": []})
    
        while len(mutated_smiles_df) < self.num_mutations:
            
            num_mutations_needed = self.num_mutations - len(mutated_smiles_df) 
            num_mutations_needed += np.ceil(0.05 * num_mutations_needed)

            parent_population = self.optimizer.select_non_elite(self.previous_generation, num_mutations_needed)

            mutated_smiles_df = self.mutator.make_mutants(generation_num=self.generation_number, num_mutants_to_make=self.num_mutations, parent_population=parent_population, new_generation_df=mutated_smiles_df)
        print("End mutation")
        return mutated_smiles_df
    
    def crossover(self):
        print("begin crossover")
        crossed_smiles_df = pd.DataFrame({"smiles": [], "parent1_id": [], "parent2_id": []})

        while len(crossed_smiles_df) < self.num_crossovers:
            
            num_crossovers_needed = self.num_crossovers - len(crossed_smiles_df) 
            num_crossovers_needed += np.ceil(0.5 * num_crossovers_needed)

            parent_population = self.optimizer.select_non_elite(self.previous_generation, num_crossovers_needed)

            crossed_smiles_df = self.CrossoverOp.make_crossovers(generation_num=self.generation_number, num_crossovers_to_make=self.num_crossovers, list_previous_gen_smiles=parent_population, new_crossover_smiles_list=crossed_smiles_df)
        print("end crossover")
        return crossed_smiles_df

    def optimize(self, population):

        self.generation_number += 1
        self.previous_generation = population

        #generate elite_pop
        elite_df = self.optimizer.select_elite_pop(population, self.num_elite)
        
        full_population = deepcopy(elite_df)
        
        while len(full_population) < self.max_population:
            #generate mutation_pop
            mutated_df = self.mutate()
            mutated_df['source'] = np.full(len(mutated_df), "mutation")
            
            #generate crossover_pop
            crossed_df = self.crossover()
            crossed_df['source'] = np.full(len(crossed_df), "crossover")
        
            #combine mutation_pop, crossover_pop, elite_pop
            full_population = pd.concat([full_population, mutated_df, crossed_df])
            full_population.reset_index(drop=True, inplace=True)

            #The following code counts how many repeated smiles strings there are. 
            # The parameter max_clones allows you to define how many repeated smiles strings can exist in the population at any given time
            smiles_counts = full_population.groupby(full_population['smiles'], as_index=False).size()
            smiles_counts = smiles_counts[smiles_counts['size'] > self.max_clones]
            smiles_to_remove = smiles_counts['smiles'].tolist()
            count = smiles_counts['size'].tolist()

            indexes_to_remove = []
            for smi, count in zip(smiles_to_remove, count):
                c = count - self.max_clones
                indexes = full_population[full_population['smiles'] == smi].tail(c).index.values.tolist()
                indexes_to_remove.extend(indexes)
            
            if len(indexes_to_remove) > 0:
                full_population.drop(indexes_to_remove, inplace=True)
                full_population.reset_index(drop=True, inplace=True)
        
        if len(full_population) > self.max_population:
            num_to_remove = abs(len(full_population) - self.max_population)
            full_population.drop(full_population.tail(num_to_remove).index, inplace=True)

        full_population['chromosome'] = np.full(len(full_population), np.nan)
        full_population.reset_index(drop=True, inplace=True)
        
        return full_population