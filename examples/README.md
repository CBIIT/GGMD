# Understanding the necessary parameters

There is an extensive list of parameters that are necessary to run this pipeline. Please refer to the below list to understand the parameters and what should be passed to understand them.

## Parameters:

- model_type: Which generative model is to be used. Current implemented options: 'jtvae' and 'autogrow
- #smiles_input_file: A string containing the file path to the smiles file that should be used as the initial population. 
- output_directory: A string containing a file path to the directory that you would like all output to be directed to. 
- scorer_type: The name of the scoring function to be used. Current implemented options: 'LogPTestCase' and 'ampl_model'
- num_epochs: The number of optimization cycles to happen before termination. This is also referred to as the number of generations. Enter a positive integer value greater than 0

#Selection/Optimization params
- optimizer_type: Which optimizer class should be used to optimize the molecules. Current implemented options: 'geneticoptimizer'
- selection_pool_size: The number of individuals that should be in each subset of molecules being considered to be parents. This value should be a positive integer that is no larger than the size of the population. 
- mate_prob: This parameter is the percentage of the population to be carried over to the next generation and be mutated and used as parents. Enter a float value between 0 and 1.
- mutate_prob: The probability of mutation occuring. This parameter determines the probability of an individual being mutated and the probability of each gene being mutated. Enter a float value between 0 and 1.
- mutation_std: This parameter is used for the mutation operator. This parameter determines the scale of the mutated genes value. Only used by the JTVAE method
- optima_type: The optima type determines what fitness values are considered good/bad. Current implemented options: "minima" and "maxima".
- selection_type: The method used to determine parent pairs for crossovers. The optimizer pulls a subset of n rows of the population at random where n is selection_pool_size. Then, the selection type selects one individual from the subset. The selection type effects how the parents are selected from the population. Current implemented options: 'roulette' and 'tournament'. 
- elite_ratio: This parameter dictates how many members will be selected as 'elite' to be directly carried over to the next generation. Elite individuals are selected based on their fitness score. The optimizer sorts the population by fitness then returns the top n rows where n is equal to elite_ratio * max_population. Value should be a float
- max_clones: This controls the number of repeated smiles strings that are allowed in the active population at any given time. Limiting to 1 means that at any given time, the population will contain no more than 1 of any smiles strings. This will help reduce pre-mature convergence. Value should be integer
- sample_with_replacement: Boolean variable indicating if selection should be done with or without replacement. Set to true means that during non-elite selection, a selected individual will be removed from the pool of individuals to select. This will result in essentially sorting the population by pairing more fit individuals together for crossover. By setting this to False, the selected individual won't be removed from the selection pool. This means that more fit individuals are likely to be selected more than once. However, this does not conflict with the max_clones parameters. If there are more repeats than the max_clones allows, repeats will be removed and selection will be repeated until population size is as expected. Additionally, the optimizer will not allow the same smiles string to be selected such that crossover will be done with both parents having the same smiles string.

- initial_pop_size: This is the size of the population in generation 0. If your smiles_input_file has more than this number, than the smiles strings from the input file will be randomly sampled down to this initial_pop_size. Enter a positive integer number greater than 0.
- max_population: Depending on your settings, the population may fluctuate in size over the generations. The max_population parameter sets a hard limit to how many individuals can be in a population. Enter a positive integer number that is greater than 0.

#FNL JTNN (These parameters are specific to the JTNN method! Only necessary if using this model type)
- vocab_path: A string containing the file path to the vocab file. Contains the vocab for the JTNN method.
- model_path: A string containing the file path to the model file. This is the PRE-TRAINED JTNN model.