import pandas as pd
import argparse, yaml
from yaml import Loader
from random import sample
import numpy as np



class Tracker():
    def __init__(self, params):
        self._next_id = 0
        self._smiles_input_file = params.smiles_input_file
        self._output_directory = params.output_directory
        self.generation = 0
        self._initial_pop_size = params.initial_pop_size

        self.master_df = pd.DataFrame(columns=['compound_id', 
                                               'smiles', 
                                               'First generation in which the molecule appears', 
                                               'Total number of generations molecule is present',
                                               'List of generations molecule is present', 
                                               'fitness', 
                                               'chromosome', 
                                               'Number of times this molecule was evolutionarily created (not carried over through elitism)',
                                               'Generation, method this molecule was created, and parent ID numbers'])

    def init_population(self):
        with open(self._smiles_input_file) as f: 
            smiles_list = [line.strip("\r\n ").split()[0] for line in f]
        
        smiles_list = sample(smiles_list, self._initial_pop_size)
        comp_ids = [i for i in range(len(smiles_list))]

        population = pd.DataFrame()
        population['compound_id'] = comp_ids
        population['smiles'] = smiles_list
        #population['generation'] = [[0] for _ in range(len(smiles_list))]
        population['source'] = ['initial' for _ in range(len(smiles_list))]
        population['parent1_id'] = np.full(len(smiles_list), np.nan)
        population['parent2_id'] = np.full(len(smiles_list), np.nan)
        population['chromosome'] = np.full(len(smiles_list), np.nan)

        self._next_id = len(comp_ids)

        #self.update_tracker(population)

        return population

    def update_tracker(self, population):

        population.reset_index(drop=True, inplace=True)

        for i in range(len(population)):
            individual = population.loc[i]
            smiles = individual['smiles']

            if smiles in self.master_df['smiles'].tolist(): #If this molecules smiles string already exists in master tracker
                index = self.master_df[self.master_df['smiles'] == smiles].index[0]
                
                if individual['source'] != 'elitism': #Created this generation but smiles has already come up before
                    population.loc[i, 'compound_id'] = self.master_df.loc[index, 'compound_id']
                    self.master_df.loc[index, 'Number of times this molecule was evolutionarily created (not carried over through elitism)'] += 1
                    self.master_df.loc[index, 'Generation, method this molecule was created, and parent ID numbers'][self.generation] = str(individual['source']) + ", " + str(individual['parent1_id']) + ", "  + str(individual['parent2_id'])
                else:
                    population.loc[i, 'compound_id'] = self._next_id
                    self._next_id += 1

                self.master_df.loc[index, 'List of generations molecule is present'].append(self.generation)
                self.master_df.loc[index, 'Total number of generations molecule is present'] += 1
            
            else: #Else: this molecules smiles string does not exists in master tracker
                if individual['source'] == 'initial': #Specific to the initial population taken from input file
                    new_row = pd.Series({
                        'compound_id': individual['compound_id'], 
                        'smiles': individual['smiles'], 
                        'First generation in which the molecule appears': self.generation,
                        'Total number of generations molecule is present': 1,
                        'List of generations molecule is present': [self.generation],
                        'fitness': individual['fitness'],
                        'chromosome': individual['chromosome'],
                        'Number of times this molecule was evolutionarily created (not carried over through elitism)': 0,
                        'Generation, method this molecule was created, and parent ID numbers': {}
                        })
                else: #Any other new molecules that are created, not introduced in the initial population
                    new_row = pd.Series({
                            'compound_id': self._next_id,
                            'smiles': individual['smiles'],
                            'First generation in which the molecule appears': self.generation,
                            'Total number of generations molecule is present': 1,
                            'List of generations molecule is present': [self.generation],
                            'fitness': individual['fitness'],
                            'chromosome': individual['chromosome'],
                            'Number of times this molecule was evolutionarily created (not carried over through elitism)': 1,
                            'Generation, method this molecule was created, and parent ID numbers': {self.generation: str(individual['source']) + ", " + str(individual['parent1_id']) + ", "  + str(individual['parent2_id'])}
                            })
                    population.loc[i, 'compound_id'] = self._next_id
                    self._next_id += 1

                self.master_df = pd.concat([self.master_df, new_row.to_frame().T], ignore_index=True)
        
        self.generation += 1
        return population
    
    def publish_data(self):
        
        self.master_df.to_csv(self._output_directory + "/data_all_generations.csv", index=False)
        print(self.master_df)

        



def test_tracker():
    # TODO: Define test for tracker

    parser = argparse.ArgumentParser()
    parser.add_argument('-config', help="Config file location *.yml", action='append', required=True)
    args = parser.parse_args()

    for conf_fname in args.config:
        with open(conf_fname, 'r') as f:
            parser.set_defaults(**yaml.load(f, Loader=Loader))

    args = parser.parse_args()
    tracker = Tracker(args)
    df = tracker.create_tracker()

    population = pd.read_csv("/mnt/projects/ATOM/blackst/GMD_workspace/ampl_ggmd_test/workspace/unscored_population.csv")

    tracker.update_tracker(population)

if __name__ == "__main__":
    test_tracker()