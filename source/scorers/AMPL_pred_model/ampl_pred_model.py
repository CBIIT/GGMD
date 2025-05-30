from scorers.base_scorer import Scorer
import pandas as pd
import os

class ampl_pred_model(Scorer):
    
    def __init__(self, params):
        self.working_directory = params.working_directory
        self.container_type = params.container_type  # 'singularity', 'docker', or 'conda'
        self.target_col_name = params.target_col_name
        
        # Conditional parameter assignments
        self.ampl_image = getattr(params, 'ampl_image', None)
        self.conda_env = getattr(params, 'conda_env', None)
        
        # Validate parameters based on container type
        if self.container_type == 'singularity' or self.container_type == 'docker':
            if not self.ampl_image:
                raise ValueError(f"ampl_image must be specified when using '{self.container_type}' as the container type.")
        
        if self.container_type == 'conda':
            if not self.conda_env:
                raise ValueError("conda_env must be specified when using 'conda' as the container type.")
        
        # Load Singularity module if using Singularity
        if self.container_type == 'singularity':
            os.system("module load singularity")
    
    def score(self, population):
        # Drop 'fitness' column if it exists
        if 'fitness' in population.columns:
            population.drop(["fitness"], axis=1, inplace=True)
        
        # Save the population to a CSV file
        population.to_csv(f"{self.working_directory}/unscored_population.csv", index=False)

        print("Executing scoring process...")
        
        # Execute scoring based on the specified container type
        if self.container_type == 'singularity':
            print("Using Singularity container...")
            os.system(f"singularity exec --bind {self.working_directory}:/data {self.ampl_image} /data/run_inference.sh")
        elif self.container_type == 'docker':
            print("Using Docker container...")
            os.system(f"docker run -v {self.working_directory}:/data {self.ampl_image} /data/run_inference.sh")
        elif self.container_type == 'conda':
            print("Using Conda environment...")
            os.system(f"mamba run -n {self.conda_env} bash -c 'cd {self.working_directory} && ./conda_run.sh'")
        else:
            raise ValueError("Invalid container_type specified. Choose 'singularity', 'docker', or 'conda'.")
        
        print("Scoring process complete.")
        
        # Read the scored population from the output CSV
        scored_population = pd.read_csv(f"{self.working_directory}/scored_population.csv")
        
        # Add fitness column to the population DataFrame
        population['fitness'] = scored_population[f'{self.target_col_name}_pred']
        
        return population
