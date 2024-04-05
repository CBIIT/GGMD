from atomsci.ddm.pipeline.predict_from_model import predict_from_model_file
import pandas as pd
import os
def main():

    model_file = "/data/model_file_name"
    population_file = "/data/unscored_population.csv"

    population = pd.read_csv(population_file)
    scored_population = predict_from_model_file(model_file, population, smiles_col="smiles", is_featurized=False, response_col='pIC50')
    scored_population.to_csv("/data/scored_population.csv", index=False)

if __name__ == "__main__": 
    main()