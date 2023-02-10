#Imports
from generative_network import create_generative_model
import argparse
import yaml
from yaml import Loader
import pandas as pd
import scorer, optimizer2
from data_tracker import Tracker

def gen_dataframe(smiles_input_file):
    with open(smiles_input_file) as f:
        smiles_list = [line.strip("\r\n ").split()[0] for line in f]
    
    df = pd.DataFrame()
    df['smiles'] = smiles_list[10000:]
    # TODO: Add a compound ID column. ID can just be a unique number. Need to figure out how to track which ID's have been used even if removed.
    # I also need to add a feature to track all unique compounds across all generations.
    return df

if __name__ == "__main__":
    #Parse args
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
    #optimizer = optimizer2.create_optimizer(args)
    generative_model = create_generative_model(args)
    tracker = Tracker(args)
    print("classes initialized")

    #initialize population
    #population = gen_dataframe(args.smiles_input_file)
    population = tracker.create_tracker()
    print("population initialized")

    #prepare population:
    population = evaluator.score(population)
    print("Population evaluated")

    #Begin optimizing
    #for epoch in range(args.num_epochs):
    for epoch in range(args.num_epochs):
        print("Inside loop")
        
        #print(population.columns)

        ############################## Optimization ##############################
        #population = optimizer.optimize(df)
        population = generative_model.optimize(population)

        population.to_csv("evaluated_pop.csv", index=False)

        #TODO: what if we tunnel optimizing through the generative model? This way,
        # the generative model can set up the molecules to then be sent on for optimization.
        # We need to encode the compounds to latent space, but that is only for latent space
        # methods. This way, we can leave it generalized for non latent-space methods to be
        # integrated.
        
        #df = optimizer.optimize(df) #Temp commented out
        ##########################################################################
        
        #evaluate population
        #df = evaluator.evaluate(population) #Temp commented out

    print("columns for population: ", population.columns)