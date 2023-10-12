This directory contains usefull scripts for preparing for using GMD and for analyzing the results. 

### JTVAE reconstruction Rate Script

If you are interested in using the JTVAE method in the GMD loop, you can either provide your own trained JTVAE model or use the provided JTVAE model. The provided model was trained on the ZINC dataset with a reconstruction rate of ~86%. Provided in this package is the trained model, vocab file, and full list of smiles strings from ZINC dataset. If you want to use a different dataset as the initial generation, we suggest running the JTVAE_reconstruction_test.py script in this directory to verify that the provided JTVAE model provides a reasonable reconstruction rate on your dataset.

If the provided model is not sufficient, you can either switch to the AutoGrow method that has been implemented or you can train a new model on your dataset. We suggest using the same source code for training a JTVAE model to ensure there are no inconsistencies in the implementation of JTVAE

JTVAE Code - https://github.com/SeanTBlack/FNL_JTVAE