This directory contains all of the source code for the GMD pipeline. 


## Python files

### main.py

This file is the main driver for the pipeline. It accepts the input config file and initializes all the other classes necessary. 

### data_tracker.py

This file maintains the master tracker file that saves off all of the data from entire GMD run. It is in charge of creating the initial generation's dataframe, defining the id numbers for each compound, keeping track of how each compound was created, and tracking which generations each compound survived in.

After all generations have been completed, the data tracker saves off a csv file containing the compounds from all generations. This will likely be a large file, so we recommend saving it to a directory that can store it. 


## Other files

There are currently three sub directories stored here:

1. generative_networks:
2. optimizers
3. scorers

### generative_networks

This directory contains the non-generalized code for the methods that change the compounds in the generation. For the latent space methods, this is where the compounds are encoded and decoded. Currently, there are two generative methods implemented: JTVAE (latent space) and AutoGrow (chemical space).

### optimizers

This directory contains the code for each of the optimizers that have been implemented. Right now, we only have one optimizer implemented: the genetic algorithm. This code is generalized and is involved in guiding the population towards a more optimized set of compounds. This code uses the previous generation and their fitness values to create the next generation. 

Any code that cannot be generalized from this section has been abstracted back to the code in the generative_networks directory. For example, using a latent space method in conjunction with the genetic algorithm requires performing mutations and crossovers on the latent vector. This cannot be generalized since different latent space methods represent their molecules in different ways. So, the genetic algorithm code performs the selection of molecules to mutate/crossover then the code in the generative_networks directory performs the mutations/crossovers and then decodes them back to smiles strings.

### scorers

This directory contains all code related to computing the fitness scores for each compound. This code has been generalized to different methods, and is the most likely section of the code to need to be edited by the user to fit your exact needs. For each research project, the goal of what you are optimizing for may change depending on your goals. Multiple methods can be combined into the one fitness score. 