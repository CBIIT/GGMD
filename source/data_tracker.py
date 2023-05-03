import pandas as pd
import argparse, yaml
from yaml import Loader
from random import sample

class Tracker():
    def __init__(self, args):
        self._next_id = 0
        self._smiles_input_file = args.smiles_input_file
        self._output_directory = args.output_directory
        self.generation = 0
        self._initial_pop_size = args.initial_pop_size

    def init_population(self):
        with open(self._smiles_input_file) as f: 
            smiles_list = [line.strip("\r\n ").split()[0] for line in f]
        
        smiles_list = sample(smiles_list, self._initial_pop_size)
        smiles_list = smiles_list[0:5]
        comp_ids = [i for i in range(len(smiles_list))]

        population = pd.DataFrame()
        population['compound_id'] = comp_ids
        population['smiles'] = smiles_list

        self._next_id = len(comp_ids)

        return population
        
    def create_tracker(self, population):

        population['generation'] = [self.generation for _ in range(len(population))]
        
        self.master_df = population

    def update_tracker(self, population):
        population.reset_index(drop=True, inplace=True)

        ids = [i for i in range(self._next_id, self._next_id + len(population))]
        population['compound_id'] = ids

        self.generation += 1
        #TODO: We should add a feature here to track all the generations that a molecule survived in.
        population['generation'] = [self.generation for _ in range(len(population))]
        
        self.master_df = pd.concat([self.master_df, population])
        self._next_id = len(self.master_df)

        population.drop(['generation', 'parent1_id', 'parent2_id'], axis=1, inplace=True)
        return population
    
    def publish_data(self):
        self.master_df.to_csv(self._output_directory + "data_all_generations.csv", index=False)


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