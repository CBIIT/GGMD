# Implemented generative methods:

## Junction Tree Variational Autoencoder (JTNN/JTVAE)

The JTVAE method is a latent space method. It functions by training an autencoder to encode a molecules SMILES string into a vector then decode that vector back into a SMILES string. When we implement the JTVAE method into the GMD pipeline, we can generate new molecules by manipulating a molecules latent vector then decoding it back into a SMILES string. 

Before using the JTVAE method, a JTVAE model must be trained. Along with this package, we have provided a trained JTVAE model. This model was trained on the ZINC dataset and exhibits a reconstruction accuracy rate of ~88%. You can run this pipeline using the SMILES strings from the ZINC dataset (also provided with this package) as the initial population. 

If you would prefer to use a different dataset, you may need to train a new JTVAE model. To determine if you need to train a new model, we recommend evaluating the reconstruction accuracy of the model on your dataset. A script to do this, as well as a sample batch script, can be found at 

```
/FNLGMD/scripts/JTVAE_reconstruction_test.py
```

If you do decide to train a new JTVAE model, is recommended to use our implementation of the JTVAE method to train a model to ensure a smooth integration of your model into the GMD pipeline. This code can be found in this repository 

https://github.com/SeanTBlack/FNL_JTVAE

Source Paper: Jin, Wengong, Regina Barzilay, and Tommi Jaakkola. "Junction tree variational autoencoder for molecular graph generation." International conference on machine learning. PMLR, 2018.

## AutoGrow

The AutoGrow method is a chemical space method. It does not use a latent vector or autencoder to represent or manipulate the molecules. Instead, the AutoGrow method manipulates molecules by simulating chemical reactions. 

AutoGrow uses SMARTS reaction notation and RDKit to perform mutations. AutoGrow comes with 94 reactions for mutations. Of these 94 reactions, 79 require two reactants. For those reactions, the first molecule is selected from the current generation and the second is selected from one of AutoGrow's complementary molecular-fragment libraries. 

AutoGrow performs crossovers by merging two compounds from the current generation. AutoGrow looks for the largest substructure that the two selected compounds have in common and "generates a child by randomly combining their decorating moieties". This is done using the RDKit library to generate the new compound via SMILES strings of the parents.

source: Spiegel, J.O., Durrant, J.D. AutoGrow4: an open-source genetic algorithm for de novo drug design and lead optimization. J Cheminform 12, 25 (2020). https://doi.org/10.1186/s13321-020-00429-4