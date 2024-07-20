#Imports
#from generative_network import create_generative_model
from generative_networks.generative_network_factory import create_generative_model
import argparse
import yaml, os, time
from yaml import Loader
from scorers.scorer_factory import create_scorer
from data_tracker import Tracker

def main(param_yaml):
    #Parse arguments
    #TODO: Need to build out a formal argument parser to help users. Should set up what parameters are 
    # required and which aren't. Then we can provide default values for some variables.
    parser = argparse.ArgumentParser()

    
    with open(param_yaml, 'r') as f:
        parser.set_defaults(**yaml.load(f, Loader=Loader))

    params = parser.parse_args()

    #initialize classes:
    evaluator = create_scorer(params)
    tracker = Tracker(params)
    print("classes initialized")

    #initialize population
    population = tracker.init_population()
    print("population initialized")
    
    generative_model = create_generative_model(params)
    population = generative_model.prepare_population(population)

    #prepare population:
    population = evaluator.score(population)
    print("Population evaluated")

    population = tracker.update_tracker(population)

    #Begin optimizing
    for epoch in range(1, params.num_epochs + 1):
        print(f"\nGeneration #{epoch}")

        population = generative_model.optimize(population)
        
        #evaluate population
        population = evaluator.score(population) 

        # Update Data Tracker
        population = tracker.update_tracker(population)
        
    tracker.publish_data()

if __name__ == "__main__":

    with open("/home/0000/003/data_prep/data5/completed6.txt", 'w') as f_out:
        yaml_files = []
        data_dir = "/home/0000/003/data_prep/data5"
        
        file_list = os.listdir(data_dir).sort()

        l = len(file_list)//6

        file_list = file_list[5*l:]

        for file in file_list:
            t0 = time.time()

            dir_path = os.path.join(data_dir, file)
            main(dir_path)
            f_out.write(file)
            
            t1 = time.time()

            print(f"File name: {file} took {t1-t0/60} minutes to process")

    