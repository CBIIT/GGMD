# About this GMD

This software is intended to act as a framework for generative molecular design research. Since this GMD follows the basic structure of 
the genetic algorithm, there are _ main components that make this pipeline work:
    
1. Scoring
2. Preperation of data
3. optimizing

For each of these componenets, this modular design allows for an easy replacement of the different implementations. The preperation of data section, for example, will work for both latent space and chemical space methods. 
The optimization section currently has two methods implemented and tested. 

If you are using a latent space implementation, this software does not currently support the process of training an autoencoder such as the JTVAE method. To use the implemented JTVAE method, please refer to that code to train
the autoencoder and generate the necessary vocabulary. 

## Installation

### Clone git repository

`git clone https://github.com/SeanTBlack/FNLGMD.git`

### Create conda environment

```
conda create -n <env_name> python=3.8
conda activate <env_name> 
conda install -c conda-forge rdkit
conda install numpy 
conda install scipy
conda install joblib
conda install networkx 
conda install pytorch torchvision cpuonly -c pytorch

pip install Pebble
```

## Getting started:

There are currently 4 parent folders in this repository that are intended to organize everything in a straight forward manner:

1. Source: This folder contains the source code that runs the pipeline.
2. Examples: This folder contains example input configuration files, output, and performance metrics to help guide you when getting started. Note that this is intended as a starting place with no guarantee that these settings will work best for a given application. Parameters will need to be experimented with to best fit your application.
3. Output: This folder is the default location for all output from the GMD pipeline will go. The output will include the final csv files containing the compounds created within the pipeline, logs, and any error messages that may have come out of the pipeline
4. Scripts: This folder contains scripts that are intended to be ran either before or after using the pipeline. These scripts include data analys and visualizations scripts with varying purpose. See further documentation to learn how to best use those scripts

An example configuration can be found at 

`workspace/example_dir`

In this foldere, there is a slurm file and configuration file that are intended to be used as a 

## Issues:

If you have any issues while using or installing this code, please create a new issue in the Issues tab of this github page.

Feedback is always greatly appreciated! This code is still in development, changes will continue to be pushed and we have many plans for future development! 