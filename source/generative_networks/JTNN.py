import time
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


class JTNN_FNL(GenerativeModel):

    def __init__(self, params):
        self.encoder = encode_smiles(params)
        self.decoder = decoder(params)
        self.is_first_epoch = True

        #New optimizer structure STB
        self.optimizer = create_optimizer(params)
        self.mate_prob = params.mate_prob
        self.max_population = params.max_population
        self.mutate_prob = params.mutate_prob
        self.mutation_std = params.mutation_std
        self.tree_sd = 4.86
        self.molg_sd = 0.0015

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
        print("Crossover beginning population size: ", len(population))
        num_children = int(self.mate_prob * self.max_population) #STB int((0.3 * 100)) = 30
        parents_idx = np.random.randint(0, len(population), (num_children, 2)) #Sets up the indexes for parents in shape [[parent1, parent2], ...]
        
        parent_1 = []
        parent_2 = []
        child_chrom = []
        for i in range(num_children):
            parents = population.iloc[parents_idx[i]]
            parent_chrom = np.vstack(parents["chromosome"].values)
            parent_1.append(parents.compound_id.values[0])
            parent_2.append(parents.compound_id.values[1])

            selected_genes = np.random.randint(0, 2, self.chromosome_length)
            child_chromosome = np.where(selected_genes, parent_chrom[1], parent_chrom[0])
            child_chrom.append(child_chromosome)

        children_df = pd.DataFrame({"chromosome": child_chrom, "fitness": np.full(num_children, np.nan), "parent1_id": parent_1, "parent2_id": parent_2})
        
        population = pd.concat([population, children_df])
        population.reset_index(drop=True, inplace=True)
        print("Number of children: ", len(children_df), " length of total population: ", len(population))
        
        return population
    
    def mutate(self, population):
        #TODO: Need to test this method. My question is that we need to decide if mutations
        # should happen in place or if a mutated individual should create a new individual in 
        # the population. Mutations happen after crossover and can happen to new or old individuals

        mut_indices = np.where(np.random.rand(len(population)) < self.mutate_prob)[0]

        for idx in mut_indices:
            chromosome = population['chromosome'].iloc[idx]

            tree_vec, mol_vec = np.hsplit(chromosome, 2)

            tree_mut_ind = np.where(np.random.rand(len(tree_vec)) < self.mutate_prob)
            tree_vec[tree_mut_ind] = np.random.normal(loc=tree_vec[tree_mut_ind], scale=self.tree_sd * self.mutation_std)
            
            mol_mut_ind = np.where(np.random.rand(len(mol_vec)) < self.mutate_prob)
            mol_vec[mol_mut_ind] = np.random.normal(loc=mol_vec[mol_mut_ind], scale=self.molg_sd * self.mutation_std)

            
            num_pts_mutated = len(tree_mut_ind) + len(mol_mut_ind)

            if num_pts_mutated > 0:
                chromosome = np.concatenate([tree_vec, mol_vec])

                if np.isnan(population['fitness'].iloc[idx]) == False:
                    # If the individuals fitness is not nan, then this is an individual
                    # from a previous generation. We need to reset fitness, smiles, parent ids
                    # and compound_id.
                    population['parent1_id'].iloc[idx] = population['compound_id'].iloc[idx]
                    population['parent2_id'].iloc[idx] = np.nan
                    population['smiles'].iloc[idx] = np.nan
                    population['fitness'].iloc[idx] = np.nan
                    population['compound_id'].iloc[idx] = np.nan

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
        
        if self.is_first_epoch:
            #Encode:
            
            smiles = population['smiles']
            chromosome = self.encode(smiles)

            assert len(smiles) == len(chromosome)

            population['chromosome'] = chromosome

            self.chromosome_length = len(population["chromosome"].iloc[0]) #When I set self.population, this can be moved potentially

            self.is_first_epoch = False

        #Optimize:
        print("Optimizing")
        size = int(self.mate_prob * len(population)) #int((1 - 0.3) * 50) = 35
        population = self.optimizer.select_non_elite(population, size)
        population = self.crossover(population=population)
        population = self.mutate(population)

        #Decode:
        smiles = self.decode(population['chromosome'].tolist())

        population['smiles'] = smiles
        
        return population
