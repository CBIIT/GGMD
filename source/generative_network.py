import pandas as pd
import numpy as np
import argparse
import logging
import yaml
import time
from yaml import Loader

import optimizer, new_optimizer

from generative_models.FNL_JTNN.fast_jtnn.gen_latent import encode_smiles, decoder
from generative_models.JTNN.VAEUtils import DistributedEvaluator

Log = logging.getLogger(__name__)

# Hack to keep molvs package from issuing debug message on bad Unicode string
molvs_log = logging.getLogger('molvs')
molvs_log.setLevel(logging.WARNING)

def create_generative_model(params):
    """
    Factory function for creating optmizer objects of the correct subclass for params.optmizer_type.
    Args:
        params: parameters to pass
    Returns:
        optmizer (object):  wrapper
    Raises: 
        ValueError: Only params.VAE_type = "JTNN" is supported
    """
    if params.model_type.lower() == "jtnn":
        return JTNN(params)
    elif params.model_type.lower() == "moses_charvae":
        print("loading moses charvae")
        return charVAE(params)
    elif params.model_type.lower() == 'jtnn-fnl':
        return JTNN_FNL(params)
    else:
        raise ValueError("Unknown model_type %s" % params.model_type)


class GenerativeModel(object):
    def __init__(self, params, **kwargs):
        """
        Initialization method for the GenerativeModel class

        Args:
            params (namespace object): contains the parameters used to initialize the class
        MJT Note: do we need the **kwargs argument?
        """
        self.params = params
    
    def optimize(self):
        """
        optimize function not implemented in super class
        """
        raise NotImplementedError



class JTNN_FNL(GenerativeModel):

    def __init__(self, params):
        self.encoder = encode_smiles(params)
        self.decoder = decoder(params)
        self.is_first_epoch = True

        #New optimizer structure STB
        self.optimizer = new_optimizer.create_optimizer(params)
        self.mate_prob = params.mate_prob
        self.max_population = params.max_population

    def encode(self, smiles):
        print("Encoding ", len(smiles), " molecules")
        chromosome = self.encoder.encode(smiles)
        
        return list(chromosome)

    def decode(self, chromosome):
        print("Decoding ", len(chromosome), " molecules")
        
        # Parallel:
        #smiles = self.decoder.decode_simple_2(chromosome)

        # Not parallel:
        smiles = self.decoder.decode_simple(chromosome)
        
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
    
    def mutation(self, population):
        #Need to implement mutations
        pass

    def sort(self, population):
        """
        This function splits the population into two sets: one set that contains the new individuals and one set that contains the unchanged individuals
        The new individuals need to be decoded, scored and added to the data tracker. The unchanged individuals have already been decoded, scored and tracked.
        This function sends the unchanged individuals to the genetic optimizer which stores the individuals until the next generation and returns the new
        individuals to be decoded, scored, and tracked
        Paramters:
            - population: dataframe of whole population
        Returns:
            - population_of_new_individuals: Pandas DataFrame of the individuals created in this generation
        """

        #Surviving individuals have real values for the fitness, smiles, and compound_id columns
        retained_population = population[population['fitness'].isna() == False]
        retained_population.drop(['parent1_id', 'parent2_id'], axis=1, inplace=True)
        self.optimizer.set_retained_population(retained_population)

        #Individuals that have been created this generation have a NaN associated with fitness, smiles, and compound_id columns
        population_of_new_individuals = population[population['compound_id'].isna()]
        return population_of_new_individuals

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

        population = self.optimizer.optimize(population)
        population = self.crossover(population=population)
        #populaiton = self.mutate(population)

        print("sort")
        population = self.sort(population)

        #Decode:
        smiles = self.decode(population['chromosome'].tolist())

        population['smiles'] = smiles
        print("Shape of population: ", population.shape)
        
        return population

def test_decoder(args):
    print("Running FNL's JTNN Test function: ")
    fname = '/mnt/projects/ATOM/blackst/GMD/LOGP-JTVAE-PAPER/Raw-Data/ZINC/all.txt'
    # Encoding time for  249456  molecules:  511.4566116333008  seconds
    # 
    #with open(args.smiles_input_file) as f:
    with open(fname) as f:
        smiles_list = [line.strip("\r\n ").split()[0] for line in f]

    smiles_original = smiles_list[:50]
    print("There are ", len(smiles_original), " molecules in the smiles list")

    model = create_generative_model(args)

    t0 = time.time()
    chromosome = model.encode(smiles_original)
    t1 = time.time()
    print("Encoding time for ", len(smiles_original), " molecules: ", (t1-t0), " seconds")
    smiles = model.decode(chromosome)
    t2 = time.time()
    print("Decoding time for ", len(smiles_original), " molecules: ", (t2-t1), " seconds")

    counter = 0
    for i in range(len(smiles_original)):
        if smiles_original[i] != smiles[i]:
            counter += 1

    print("Number of smiles incorrectly decoded: ", counter, " Reconstruction error: ", 100*(counter / (len(smiles_original))), "%")



class JTNN(GenerativeModel):

    def __init__(self, params):
        self.device = params.device
        self.vae_path = params.vae_path
        self.vocab_path = params.vocab_path

        self.vae = DistributedEvaluator(
            #device=self.device,
            timeout=15, #TODO: is this a parameter likely to be tweaked?
            vae=self.vae_path,
            vocab=self.vocab_path,
        )

        self.optimizer = optimizer2.create_optimizer(params)

    def encode(self, population):
        smiles = population['smiles'].tolist()
        chromosome, keep_compounds, dataset = self.vae.encode_smiles(smiles) #TODO: do we need this returned variable: dataset???
        print("smiles encoded")

        population = population.iloc[keep_compounds]
        population['chromosome'] = list(chromosome)
        #TODO: Look into this warning:
        #/mnt/projects/ATOM/blackst/GenGMD/source/generative_network.py:149: SettingWithCopyWarning: 
        #A value is trying to be set on a copy of a slice from a DataFrame.
        #Try using .loc[row_indexer,col_indexer] = value instead
        
        return population
    
    def encode_test(self, smiles):
        # TODO: This is a test function to test the idea of only sending the compounds
        # that have not previously been encoded. This would save time over generations. 
        # Need to continue to explore how to handle removed compounds.

        chromosome, keep_compounds, dataset = self.vae.encode_smiles(smiles) #TODO: do we need this returned variable: dataset???

        return smiles, chromosome
        
    def decode(self, chromosome):

        if len(chromosome) == 0:
            raise Exception("the chromosome seems emtpy...")

        if type(chromosome) is list:
            chromosome = [np.asarray(l) for l in chromosome]
        else:
            print(type(chromosome))

        smiles, _ = self.vae.decode_smiles(chromosome)
        
        return smiles

    def optimize_test(self, population):
        # TODO: This is a test function to test the idea of only sending the compounds
        # that have not previously been encoded. This would save time over generations. 
        # Need to continue to explore how to handle removed compounds.

        smiles_to_encode = population['smiles'].loc[population['chromosome'].isna()]

        chromosome = self.encode_test(smiles_to_encode)

        for s, l in zip(smiles_to_encode, chromosome):
            population['chromosome'].loc[population['smiles'] == s] = l

        population = self.optimizer.optimize(population)
    
    def optimize(self, population):
        population = self.encode(population)

        population = self.optimizer.optimize(population)


        chromosome = list(population['chromosome'].loc[population['smiles'].isna()])

        print(type(chromosome))
        smiles = self.decode(chromosome)
        for s, l in zip(smiles, chromosome):
            population['smiles'].loc[population['chromosome'] == l] = s

        #TODO: Write test functions for 

        return population



class CHAR_VAE(GenerativeModel):

    def __init__(self, params):
        pass
    def encode(self):
        pass
    def decode(self):
        pass
    def optimize(self):
        pass


def test_jtvae(args):
    population = pd.read_csv("/mnt/projects/ATOM/blackst/FNLGMD/source/evaluated_pop.csv")
    chromosome = list(population['chromosome'].loc[population['smiles'].isna()])

    print(type(chromosome))
    print(type(chromosome[0]))
    #print(type(np.asarray(chromosome)))
    #print(len(chromosome))
    #print(len(chromosome[0]))

    vae = create_generative_model(args)

    smiles = vae.decode(chromosome)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', help="Config file location *.yml", action='append', required=True)
    args = parser.parse_args()

    for conf_fname in args.config:
        with open(conf_fname, 'r') as f:
            parser.set_defaults(**yaml.load(f, Loader=Loader))

    args = parser.parse_args()

    test_decoder(args)
    #test_jtvae(args)










