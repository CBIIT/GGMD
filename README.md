# About this GMD

This software is intended to act as a framework for generative molecular design research. Since this GMD follows the basic structure of 
the genetic algorithm, there are 3 main components that make this pipeline work:
    
1. Scoring
2. Preperation of data
3. optimizing

For each of these componenets, this modular design allows for an easy replacement of each component. 
The optimization section currently has one method implemented and tested, the genetic optimizer.

If you are using a latent space implementation, this software does not currently support the process of training an autoencoder such as the JTVAE method. To use the implemented JTVAE method, please refer to that code to train
the autoencoder and generate the necessary vocabulary. 

Any new method generative model must be implemented as a class in the generative_network.py file. 

## Installation

### Clone git repository

`git clone https://github.com/SeanTBlack/FNLGMD.git`

### Create conda environment

```
conda create -n <env_name> python=3.8
conda activate <env_name> 
conda install -c conda-forge rdkit=2020.09.5 
conda install numpy 
conda install scipy
conda install joblib
conda install networkx 
conda install pytorch torchvision cpuonly -c pytorch

pip install Pebble
```

## Getting started:

There are currently 4 parent folders in this repository that are intended to organize this repository:

1. Source: This folder contains the source code that runs the pipeline.
2. Examples: This folder contains example input configuration files, output, and performance metrics to help guide you when getting started. Note that this is intended as a starting place with no guarantee that these settings will work best for a given application. Parameters will need to be experimented with to best fit your application.
3. Output: This folder is the default location for all output from the GMD pipeline will go. The output will include the final csv files containing the compounds created within the pipeline, logs, and any error messages that may have come out of the pipeline
4. Scripts: This folder contains scripts that are intended to be ran either before or after using the pipeline. These scripts include data analys and visualizations scripts with varying purpose. See further documentation to learn how to best use those scripts

An example configuration can be found at 

`workspace/example_dir`

In this foldere, there is a slurm file and configuration file that are intended to be used as a 

There is no "go to" values for many of the parameters. Each problem requires different settings and there is not quick way to determine the optimal values. Changing the parameters can have significant effects on the selection pressure, convergence, and diversity. It is important that you explore these parameters and learn about the effect each parameter has on the population as a whole. 

The genetic algorithm is just a starting point. There are many modifications to canonical GA that can be done for particular applications. Consider trying new variations as well such as changing the selection strategy, introducing elitism into the population, or creating an adaptive GA where the mutation and crossover probabilities change depending on performance. This repository is intended to be modified and tailored to a specific use case. We will introduce some educational material into the repository in a later update to help guide new users in how to change parameters.

## Issues:

If you have any issues while using or installing this code, please create a new issue in the Issues tab of this github page. Someone will respond to the issue as soon as possible.

Feedback is always greatly appreciated! This code is still in development, changes will continue to be pushed and we have many plans for future development! 

## Development:

If you want to make changes to the code base, please create a new branch from the master branch then create a merge request. Changes can include, but are not limited to implementing new generative models, scoring functions, optimization methods, etc...

