import time, copy 
import logging
import numpy as np
import pandas as pd
from optimizers.optimizer_factory import create_optimizer
from generative_networks.base_generative_model import GenerativeModel
from generative_networks.generative_models_helpers.FNL_JTNN.fast_jtnn.gen_latent import encode_smiles, decoder

Log = logging.getLogger(__name__)

# Hack to keep molvs package from issuing debug message on bad Unicode string
molvs_log = logging.getLogger('molvs')
molvs_log.setLevel(logging.WARNING)

pd.options.mode.chained_assignment = None  # This line silences some of the pandas warning messages

class JTNN_FNL(GenerativeModel):

    def __init__(self, params):
        self.encoder = encode_smiles(params)
        self.decoder = decoder(params)
        self.is_first_epoch = True

        #New optimizer structure STB
        self.optimizer = create_optimizer(params)
        self.max_population = params.max_population
        self.mutate_prob = params.mutate_prob
        self.mutation_std = params.mutation_std

        self.elite_size = int(params.elite_ratio * self.max_population)
        if self.elite_size % 2 != 0:
            self.elite_size += 1
        self.non_elite_size = int(self.max_population - self.elite_size)

        self.max_clones = params.max_clones

        #population = self.encode_first_generation(population)

    def prepare_population(self, population):
    
        #Encode:
        smiles = population['smiles'].to_list()
        chromosome = self.encode(smiles)

        assert len(smiles) == len(chromosome)

        population['chromosome'] = chromosome

        self.chromosome_length = len(population["chromosome"].iloc[0]) #When I set self.population, this can be moved potentially
        
        return population
    
    def encode(self, smiles) -> list:
        print("Encoding ", len(smiles), " molecules")
        
        t1 = time.time()

        chromosome = self.encoder.encode(smiles)
        
        t2 = time.time()
        print("Encoding took ", t2-t1, " seconds")

        return list(chromosome)

    def decode(self, chromosome) -> list:
        print("Decoding ", len(chromosome), " molecules")

        t1 = time.time()
        
        # Parallel:
        smiles = self.decoder.decode_simple_2(chromosome)

        # Not parallel:
        #smiles = self.decoder.decode_simple(chromosome)

        t2 = time.time()

        print("Decoding took ", t2-t1, " seconds")

        return smiles
        
    def crossover(self, population):
        num_children = len(population)

        parent_1 = []
        parent_2 = []
        child_chrom = []

        for i in range(0, len(population), 2):
            parent1 = population.iloc[i]
            parent2 = population.iloc[i+1]

            parent1_chrom = parent1["chromosome"]
            parent2_chrom = parent2["chromosome"]

            #Add in parent_1 and parent_2 id's for child1 since child1 is being added to the list first later.
            parent_1.append(parent1.compound_id)
            parent_2.append(parent2.compound_id)

            #Now add in parent_1 and parent_2 id's for child2. These id's should be flipped from child1's parents.
            parent_1.append(parent2.compound_id)
            parent_2.append(parent1.compound_id)

            try:
                parent1_tree_vec_left, parent1_tree_vec_right, parent1_mol_vec_left, parent1_mol_vec_right = np.hsplit(parent1_chrom, 4)
                parent2_tree_vec_left, parent2_tree_vec_right, parent2_mol_vec_left, parent2_mol_vec_right = np.hsplit(parent2_chrom, 4)
            except:
                parent1_tree_vec, parent1_mol_vec = np.hsplit(parent1_chrom, 2)
                parent2_tree_vec, parent2_mol_vec = np.hsplit(parent2_chrom, 2)

                parent1_tree_vec_left, parent1_tree_vec_right = parent1_tree_vec[0:int(len(parent1_tree_vec))], parent1_tree_vec[int(len(parent1_tree_vec)):]
                parent2_tree_vec_left, parent2_tree_vec_right = parent2_tree_vec[0:int(len(parent2_tree_vec))], parent2_tree_vec[int(len(parent2_tree_vec)):]

                parent1_mol_vec_left, parent1_mol_vec_right = parent1_mol_vec[0:int(len(parent1_mol_vec))], parent1_mol_vec[int(len(parent1_mol_vec)):]
                parent2_mol_vec_left, parent2_mol_vec_right = parent2_mol_vec[0:int(len(parent2_mol_vec))], parent2_mol_vec[int(len(parent2_mol_vec)):]

            child1_chrom = np.concatenate([parent1_tree_vec_left, parent2_tree_vec_right, parent1_mol_vec_left, parent2_mol_vec_right])
            child2_chrom = np.concatenate([parent2_tree_vec_left, parent1_tree_vec_right, parent2_mol_vec_left, parent1_mol_vec_right])
            
            child_chrom.append(child1_chrom)
            child_chrom.append(child2_chrom)
        
        children_df = pd.DataFrame({"compound_id": np.full(num_children, np.nan), "chromosome": child_chrom, "fitness": np.full(num_children, np.nan), "source": np.full(num_children, "crossover"), "parent1_id": parent_1, "parent2_id": parent_2})
        
        return children_df
    
    def mutate(self, population):
       
        mut_indices = np.where(np.random.rand(len(population)) < self.mutate_prob)[0]

        for idx in mut_indices:
            chromosome = population['chromosome'].iloc[idx]

            tree_vec, mol_vec = np.hsplit(chromosome, 2)

            tree_mut_ind = np.random.randint(0, len(tree_vec), 1)
            tree_vec[tree_mut_ind] = np.random.normal(loc=tree_vec[tree_mut_ind], scale=self.mutation_std) #TODO: Is this the right way to handle the scale of mutations?
            
            mol_mut_ind = np.random.randint(0, len(mol_vec), 1)
            mol_vec[mol_mut_ind] = np.random.normal(loc=mol_vec[mol_mut_ind], scale=self.mutation_std) #TODO: Is this the right way to handle the scale of mutations?
            
            
            num_pts_mutated = len(tree_mut_ind) + len(mol_mut_ind)

            if num_pts_mutated > 0:
                chromosome = np.concatenate([tree_vec, mol_vec])

                if population['source'].iloc[idx] == "crossover":
                    population['source'].iloc[idx] = "crossover + mutation"
                else:
                    population['source'].iloc[idx] = "mutation"

                population['chromosome'].iloc[idx] = chromosome

        return population

    def optimize(self, population):
        """
        This is the function responsible for handling all tasks related to optimization. For JTVAE, this includes encoding, 
        then sending the latent vectors (in the form of a pandas dataframe) to the genetic optimizer code. Once the population
        is returned, the latent vectors are decoded to SMILES strings. The resulting population is returned to the main loop 
        for scoring.

        Arguments:
        population - Pandas dataframe containing columns for id, smiles, and fitness

        Returns: 
        population - Pandas dataframe containing new smiles strings, ids, and 
        """
        
        """if self.is_first_epoch:
            #Encode:
            population = self.encode_first_generation(population)"""

        #Optimize:
        print("Optimizing")

        #Elite population:
        elite_population = self.optimizer.select_elite_pop(population, self.elite_size)
        elite_population['source'] = np.full(len(elite_population), 'elitism')

        #Non-elite population:
        full_population = copy.deepcopy(elite_population)
        #print("Before: population.shape ", population.shape)
        #print(population['fitness'].to_list())
        #print("population.columns", population.columns)

        while len(full_population) < self.max_population:
            #num_needed can be increased to create an excess of children to help in the case of duplicated smiles strings
            num_needed = int((self.max_population - len(full_population)) * 1.5) 
            if num_needed % 2 != 0: #num_needed needs to be an even number to select an even number of parents for reproduction
                num_needed += 1

            #Create next generation
            next_batch = self.optimizer.select_non_elite(population, num_needed)
            next_batch = self.crossover(next_batch)
            next_batch = self.mutate(next_batch)
            
            #Decode new molecules
            next_batch_smiles = self.decode(next_batch['chromosome'].tolist())
            next_batch['smiles'] = next_batch_smiles

            full_population = pd.concat([full_population, next_batch])

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
            
            full_population.drop(indexes_to_remove, inplace=True)
            

        if len(full_population) > self.max_population:
            num_to_remove = abs(len(full_population) - self.max_population)
            full_population.drop(full_population.tail(num_to_remove).index, inplace=True)
            print(f"Removing {num_to_remove} rows due to excess members of population.")
        
        full_population.reset_index(drop=True, inplace=True)
        return full_population
