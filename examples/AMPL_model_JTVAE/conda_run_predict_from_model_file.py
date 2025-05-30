from atomsci.ddm.pipeline.predict_from_model import predict_from_model_file
import pandas as pd
import os
def main():

    model_file = f"merged_data_model_30ae49d5-eca0-4fe2-8e49-05da8195f1b0.tar.gz"
    #model_file = "/data/bd1_bindingdb_curated_model_0c34da00-96e0-44f0-bb18-40ef64c5e3bc.tar.gz"
    population_file = f"unscored_population.csv"

    population = pd.read_csv(population_file)
    scored_population = predict_from_model_file(model_file, population, smiles_col='smiles', is_featurized=False, response_col='avg_pic50')
    scored_population.to_csv("scored_population.csv", index=False)

if __name__ == "__main__": 
    main()
