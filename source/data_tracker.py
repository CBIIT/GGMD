import pandas as pd
import argparse, yaml
from yaml import Loader
from random import sample

class Tracker():
    
    def __init__(self, args):
        self.next_id = 0
        self.smiles_input_file = args.smiles_input_file
        self.output_directory = args.output_directory
        self.generation = 0
        self.initial_pop_size = args.initial_pop_size

    def init_population(self):
        with open(self.smiles_input_file) as f: 
            smiles_list = [line.strip("\r\n ").split()[0] for line in f]
        
        smiles_list = sample(smiles_list, self.initial_pop_size)
        comp_ids = [i for i in range(len(smiles_list))] #TODO: Do we need a better compound id system?

        population = pd.DataFrame()
        population['compound_id'] = comp_ids
        population['smiles'] = smiles_list

        self.next_id = len(comp_ids)

        return population
        
    def create_tracker(self, population):

        population['generation'] = [self.generation for _ in range(len(population))]
        
        self.master_df = population

    def update_tracker(self, population):
        population.reset_index(drop=True, inplace=True)
        print(population)

        ids = [i for i in range(self.next_id, self.next_id + len(population))]
        population['compound_id'] = ids

        self.generation += 1
        population['generation'] = [self.generation for _ in range(len(population))]
        
        self.master_df = pd.concat([self.master_df, population])
        self.next_id = len(self.master_df)

        population.drop(['generation', 'parent1_id', 'parent2_id'], axis=1)
        return population
    
    def publish_data(self):

        self.master_df.to_csv(self.output_directory + "data_all_generations.csv", index=False)


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