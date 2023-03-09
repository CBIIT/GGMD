print("Pre-imports")
#Imports
from generative_network import create_generative_model
import argparse
import yaml
from yaml import Loader
import pandas as pd
import scorer, optimizer
from data_tracker import Tracker

"""
def gen_dataframe(smiles_input_file):
    with open(smiles_input_file) as f:
        smiles_list = [line.strip("\r\n ").split()[0] for line in f]
    
    df = pd.DataFrame()
    df['smiles'] = smiles_list[10000:]
    # TODO: Add a compound ID column. ID can just be a unique number. Need to figure out how to track which ID's have been used even if removed.
    # I also need to add a feature to track all unique compounds across all generations.
    return df
"""
print("Outside functions")
def main():
    print("Before parser")
    #Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', help="Config file location *.yml", action='append', required=True)
    args = parser.parse_args()

    for conf_fname in args.config:
        with open(conf_fname, 'r') as f:
            parser.set_defaults(**yaml.load(f, Loader=Loader))

    args = parser.parse_args()
    print("args: ", args)

    #initialize classes:
    evaluator = scorer.create_scorer(args)
    generative_model = create_generative_model(args) #TEMP COMMENT
    tracker = Tracker(args)
    print("classes initialized")

    #initialize population
    population = tracker.create_tracker()
    print("population initialized")

    #prepare population:
    population = evaluator.score(population)
    print("Population evaluated")

    #Begin optimizing
    #for epoch in range(args.num_epochs):
    for epoch in range(args.num_epochs):
        print(f"epoch #{epoch}")
        print("Inside loop")
        
        population = generative_model.optimize(population) #TEMP COMMENT

        population.to_csv(args.output_directory + "optimized_pop.csv", index=False)
        
        #evaluate population
        #population = evaluator.score(population) 

        population.to_csv(args.output_directory + "evaluated_pop.csv", index=False)

        # Update Data Tracker
        #tracker.update_tracker(population)

    #print("columns for population: ", population.columns)

if __name__ == "__main__":
    print("THIS IS A TEST: Commented out decoder AND optimizer call in gen_net.py optimize")
    main()
