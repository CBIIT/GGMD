import pandas as pd

class Tracker():
    
    def __init__(self, args):
        self.next_id = 0
        self.smiles_input_file = args.smiles_input_file
        
    def create_tracker(self):
        with open(self.smiles_input_file) as f:
            smiles_list = [line.strip("\r\n ").split()[0] for line in f]
        
        df = pd.DataFrame({'compound_id': [], 'smiles': [], 'latent': []})

        comp_ids = [i for i in range(len(smiles_list))]
        df['compound_id'] = comp_ids
        df['smiles'] = smiles_list
        

        df.to_csv("data_all_generations.csv", index=False)
        
        # TODO: Add a compound ID column. ID can just be a unique number. Need to figure out how to track which ID's have been used even if removed.
        # I also need to add a feature to track all unique compounds across all generations.

        self.next_id = len(comp_ids)

        return df

    def update_tracker(self):
        
        df = pd.read_csv("data_all_generations.csv")


def test_tracker():
    # TODO: Define test for tracker

    smiles_file = '/mnt/projects/ATOM/blackst/GenGMD/source/generative_models/JTNN/icml18_jtnn/data/moses/debug.txt'
    

    args = []

    tracker = Tracker(args) 

    df = tracker.create_tracker(smiles_file)
    
    assert tracker.next_id == 500

    assert list(df.columns) == ['compound_id', 'smiles', 'latent']

    
if __name__ == "__main__":
    test_tracker()
