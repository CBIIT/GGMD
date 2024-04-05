# About this example

This data came from the original JTVAE paper

This data is being used as a test case to validate the GMD loop and as a comparison test case to compare various methods for optimization and preparing the data (autoencoders or not).

This example requires a few files and all of which have been provided in this directory. Below is a brief description of each file and it's importance in the GMD loop:

- all_vocab.txt: This file is a text file of the vocabulary fragments that have been extracted from the SMILES strings located in the all.txt file. The JTVAE utilizes the vocabulary as building blocks to piece together the molecular structure during encoding and decoding.
- all.txt: This is a list of SMILES strings that all come from the ZINC library. This dataset was used to define the vocabulary for the JTVAE model and is the dataset that the provided JTVAE model was trained on. We use this file to provide a set of SMILES strings to sample from as the initial population for the GMD loop.
- config.yaml: This yaml file contains all of the user defined variables necessary for the GMD loop. Some variables are generic to the GMD loop while others are specific to the scorer, optimizer, and generative method being used. To see a comprehensive description of each of these variables please view the README file at `examples/README.md`
- model.epoch-35 is the trained JTVAE model that we have provided for this example. This model was trained using the SMILES strings from the all.txt file and the vocabulary in the all_vocab.txt file. *NOTE*: before you use this model on any other dataset than the one provided in all.txt, it is recommended to verify that this model will work on your dataset. To do so, we have provided a script that computes the reconstruction accuracy on a given dataset. Information on this can be found in the following README file: `scripts/README.md`. If this model won't work for your dataset, you can train your own JTVAE model using our code in the following repository: [`https://github.com/CBIIT/JTVAE`](https://github.com/CBIIT/JTVAE)
- Slurm-Run_Loop: This is an example slurm script provided for your reference to initialize the GMD loop for this example.