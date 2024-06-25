from scorers.base_scorer import Scorer
import pandas as pd
import os

class ampl_pred_model(Scorer):
    
    def __init__(self, params):
        self.working_directory = params.working_directory
        self.ampl_image = params.ampl_image
        self.container_type = params.container_type
        if self.container_type == 'singularity':
            os.system("module load singularity")
        self.target_col_name = params.target_col_name

    def score(self, population):

        #Created the singularity container for this using the following command:
        #       singularity pull ampl.sif docker://atomsci/atomsci-ampl:v1.5.0
        # This command pulls AMPL's official docker container and the generates a singularity container from it

        if 'fitness' in population.columns:
            population.drop(["fitness"], axis=1, inplace=True)

        population.to_csv(f"{self.working_directory}/unscored_population.csv", index=False)

        print("entering container")
        if self.container_type == 'singularity':
            os.system(f"singularity exec --bind {self.working_directory}:/data {self.ampl_image} /data/run_inference.sh")
        elif self.container_type == 'docker':
            os.system(f"docker run -v {self.working_directory}:/data {self.ampl_image} /data/run_inference.sh")
        print("container complete")

        scored_population = pd.read_csv(f"{self.working_directory}/scored_population.csv")

        #Add fitness from scored_population to population
        population['fitness'] = scored_population[f'{self.target_col_name}_pred']

        return population
