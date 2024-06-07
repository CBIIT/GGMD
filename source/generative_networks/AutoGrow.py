import numpy as np
import pandas as pd
from optimizers.optimizer_factory import create_optimizer
from generative_networks.base_generative_model import GenerativeModel
from generative_networks.generative_models_helpers.AutoGrow.mutation.execute_mutations import Mutator
from generative_networks.generative_models_helpers.AutoGrow.crossover.execute_crossover import CrossoverOp

class AutoGrow(GenerativeModel):
    def __init__(self, params):
        self.num_crossovers = params.num_crossovers
        self.num_mutations = params.num_mutations
        self.num_elite = params.num_elite
        
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
        
        #generate mutation_pop
        mutated_df = self.mutate()
        source = ['mutation' for _ in range(len(mutated_df))]
        mutated_df['source'] = source
        mutated_df['generation'] = [[] for _ in range(len(mutated_df))]
        
        #generate crossover_pop
        crossed_df = self.crossover()
        source = ['crossover' for _ in range(len(crossed_df))]
        crossed_df['source'] = source
        crossed_df['generation'] = [[] for _ in range(len(crossed_df))]
        
        #generate elite_pop
        elite_df = self.optimizer.select_elite_pop(population, self.num_elite)
        print(f"Elite population: size {elite_df.shape}")
        print("\n")
        
        #combine mutation_pop, crossover_pop, elite_pop
        combined_population = pd.concat([mutated_df, crossed_df, elite_df])
        combined_population.reset_index(drop=True, inplace=True)
        
        return combined_population
