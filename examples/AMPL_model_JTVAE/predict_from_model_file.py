from atomsci.ddm.pipeline.predict_from_model import predict_from_model_file
import pandas as pd
import os
def main():

    model_file = f"/data/TDP1_curated_model_4446f130-56b6-4d3d-96b6-2f5df715fff1.tar.gz"
    population_file = f"/data/unscored_population.csv"

    population = pd.read_csv(population_file)
    scored_population = predict_from_model_file(model_file, population, smiles_col="smiles", is_featurized=False, response_col='pIC50')
    scored_population.to_csv("/data/scored_population.csv", index=False)

if __name__ == "__main__": 
    main()