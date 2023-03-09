import pandas as pd
import argparse, yaml
from yaml import Loader

class Tracker():
    
    def __init__(self, args):
        self.next_id = 0
        self.smiles_input_file = args.smiles_input_file
        self.output_directory = args.output_directory
        
    def create_tracker(self):
        with open(self.smiles_input_file) as f:
            smiles_list = [line.strip("\r\n ").split()[0] for line in f]
        smiles_list = smiles_list[:50]
        df = pd.DataFrame({'compound_id': [], 'smiles': [], 'latent': []})

        comp_ids = [i for i in range(len(smiles_list))] #Need better compound ID
        df['compound_id'] = comp_ids
        df['smiles'] = smiles_list
        print(len(smiles_list), " compounds in the df")
        df.to_csv(self.output_directory + "data_all_generations.csv", index=False)
        
        # TODO: Add a compound ID column. ID can just be a unique number. Need to figure out how to track which ID's have been used even if removed.
        # I also need to add a feature to track all unique compounds across all generations.

        self.next_id = len(comp_ids)

        return df

    def update_tracker(self, population):
        #Would it be helpful to know which 
        
        master_df = pd.read_csv(self.output_directory + "data_all_generations.csv")

        combined_df = pd.concat([master_df, population]).drop_duplicates(subset='compound_id', ignore_index=True)
        self.next_id = len(combined_df)

        combined_df.to_csv(self.output_directory + "data_all_generations.csv", index=False)



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
    
    assert tracker.next_id == 500
    assert list(df.columns) == ['compound_id', 'smiles', 'latent']
    assert len(df) == 500

    
if __name__ == "__main__":
    test_tracker()