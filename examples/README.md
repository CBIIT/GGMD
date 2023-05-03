# Understanding the necessary parameters

There is an extensive list of parameters that are necessary to run this pipeline. Please refer to the below list to understand the parameters and what should be passed to understand them.

## Parameters:

- model_type: Which generative model is to be used. Current implemented options: 'jtnn-fnl'
- #smiles_input_file: A string containing the file path to the smiles file that should be used as the initial population. 
- output_directory: A string containing a file path to the directory that you would like all output to be directed to. 
- scorer_type: The name of the scoring function to be used. Current implemented options: 'FNL'
- num_epochs: The number of optimization cycles to happen before termination. This is also referred to as the number of generations. Enter a positive integer value greater than 0

#Selection/Optimization params
- optimizer_type: Which optimizer class should be used to optimize the molecules. Current implemented options: 'geneticoptimizer'
- tourn_size: The number of individuals that should be in each tournament if using tournament selection. This value should be a positive integer that is no larger than the size of the population.
- mate_prob: This parameter is the percentage of the population to be carried over to the next generation and be mutated and used as parents. Enter a float value between 0 and 1.
- mutate_prob: The probability of mutation occuring. This parameter determines the probability of an individual being mutated and the probability of each gene being mutated. Enter a float value between 0 and 1.
- mutation_std: This parameter is used for the mutation operator. This parameter determines the scale of the mutated genes value.
- optima_type: The optima type determines what fitness values are considered good/bad. Current implemented options: "minima" and "maxima".
- selection_type: Enter which selection method should be used. The selection type effects how the parents are selected from the population. 

- initial_pop_size: This is the size of the population in generation 0. If your smiles_input_file has more than this number, than the smiles strings from the input file will be randomly sampled down to this initial_pop_size. Enter a positive integer number greater than 0.
- max_population: Depending on your settings, the population may fluctuate in size over the generations. The max_population parameter sets a hard limit to how many individuals can be in a population. Enter a positive integer number that is greater than 0.

#FNL JTNN (These parameters are specific to the JTNN method! Only necessary if using this model type)
- vocab_path: A string containing the file path to the vocab file. Contains the vocab for the JTNN method.
- model_path: A string containing the file path to the model file. This is the PRE-TRAINED JTNN model.