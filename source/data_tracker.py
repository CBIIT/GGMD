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

        self.master_df = pd.DataFrame()

    def init_population(self):
        with open(self._smiles_input_file) as f: 
            smiles_list = [line.strip("\r\n ").split()[0] for line in f]
        
        smiles_list = sample(smiles_list, self._initial_pop_size)
        comp_ids = [i for i in range(len(smiles_list))]

        population = pd.DataFrame()
        population['compound_id'] = comp_ids
        population['smiles'] = smiles_list
        population['generation'] = [[0] for _ in range(len(smiles_list))]
        population['source'] = ['initial' for _ in range(len(smiles_list))]
        population['parent1_id'] = np.full(len(smiles_list), np.nan) #NEW
        population['parent2_id'] = np.full(len(smiles_list), np.nan) #NEW

        self._next_id = len(comp_ids)

        return population

    def update_tracker(self, population):
        population.reset_index(drop=True, inplace=True)

        #Set ID numbers for the new individuals
        uids = population['compound_id'].tolist()
        counter = 0
        for i in range(len(uids)):
            if not np.isnan(uids[i]):
                continue
            uids[i] = self._next_id + counter
            counter += 1
        population['compound_id'] = uids

        #Update generation values
        self.generation += 1
        generations = population['generation']
        for i in range(len(generations)):
            if type(generations[i]) is not list:
                generations[i] = [self.generation]
            else:
                generations[i].append(self.generation)

        population['generation'] = generations

        """
        df_new = population.iloc[np.where(population['compound_id'].isna())]
        ids, gen = [], []
        for x in range(len(df_new)):
            ids.append(50 + x)
            gen.append([1])
        #ids = [50 + x for x in range(len(df_new))]
        #gen = [[1] for _ in range(len(df_new))]
        population['compound_id'].iloc[np.where(population['compound_id'].isna())] = ids
        population['generation'].iloc[np.where(population['generation'].isna())] = gen

        df_old = population.iloc[np.where(population['generation'].isna() != True)]
        generations = df_old['generation'].tolist()
        for g in gens:
            g.append(self.generation)
        """
        
        #Add current generation's population to the master tracker
        self.master_df = pd.concat([self.master_df, population])
        self._next_id = len(self.master_df)

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

    population = pd.read_csv("/mnt/projects/ATOM/blackst/FNLGMD/workspace/example_dir/optimized_pop.csv")

    tracker.update_tracker(population)



if __name__ == "__main__":
    test_tracker()