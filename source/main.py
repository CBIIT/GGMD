#Imports
from generative_network import create_generative_model
import argparse
import yaml
from yaml import Loader
import pandas as pd
import scorer, optimizer
from data_tracker import Tracker


def main():
    #Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', help="Config file location *.yml", action='append', required=True)
    args = parser.parse_args()

    for conf_fname in args.config:
        with open(conf_fname, 'r') as f:
            parser.set_defaults(**yaml.load(f, Loader=Loader))

    args = parser.parse_args()

    #initialize classes:
    evaluator = scorer.create_scorer(args)
    generative_model = create_generative_model(args) #TEMP COMMENT
    tracker = Tracker(args)
    print("classes initialized")

    #initialize population
    population = tracker.init_population()
    print("population initialized")

    #prepare population:
    population = evaluator.score(population)
    tracker.create_tracker(population)
    print("Population evaluated")

    #Begin optimizing
    for epoch in range(args.num_epochs):
        print(f"epoch #{epoch}")
        
        population = generative_model.optimize(population)
        
        #evaluate population
        population = evaluator.score(population) 

        #print(population)

        # Update Data Tracker
        population = tracker.update_tracker(population)

    print("columns for population: ", population.columns)
    tracker.publish_data()

if __name__ == "__main__":
    main()
    #TODO: Something is happening with scoring function where sometimes it returns nan and other times it returns values. 
    # Nan when cycle == 0