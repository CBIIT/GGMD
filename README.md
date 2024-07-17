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

There are two methods to execute this GMD pipeline: direct install or using a container. It is highly recommended to utilize the container rather than direct install to minimize errors. This pipeline requires several python packages that each have particular dependencies. The containers simplify the install process and increases reproducibility in results. 


### Direct install: Create conda environment

```
git clone https://github.com/CBIIT/GGMD.git
cd GGMD
conda create --name gmd_env --file spec-file.txt
```


### Container execution: 

There are two containers for this software: one built with Singularity and one built with Docker

 Singularity is a container management software similar to Docker. Due to certain features of Docker, it is often banned on most HPC systems. Singularity was built to serve as a replacement for Docker on HPC systems while still accepting Docker containers. Due to the high computational requirements for running GMD, we expect that most users will be using HPC systems or cloud computing systems. Most HPC systems will have singularity installed. Contact your system administrators for assistance.

#### Singularity:

To install the singularity image, run the below command:

`singularity pull library://seantblack/gmd/gmd_0_9.sif`

Below are steps to run GMD through the singularity container.

```
Set up a working directory (in the following steps, replace <dir> with that directory path) with read/write permission
Copy contents of FNLGMD/workspace/LogP_demo into <dir>
Edit the output_directory parameter in the config.yaml file that is now in <dir> to be 
    output_directory: '<dir>/'
$ singularity exec --bind /<dir>:/data /path/to/gmd_img.sif /run_gmd.sh
```

If you receive the following error, your system administrators may have limited singularities access to write in your working directory `<dir>`. Try setting up your working directory in a different directory location such as your home directory.

```
OSError: Cannot save file into a non-existent directory: '<dir>'
```

#### Docker:

To install the Docker image, run the below command:

`docker pull seantaylorblack/ggmd:1.1`

Below are steps to run GMD through the Docker container.

```
- Set up a working directory (in the following steps, replace <dir> with that directory path) with read/write permission
- Copy contents of FNLGMD/workspace/LogP_demo into <dir>
- Edit the output_directory parameter in the config.yaml file that is now in <dir> to be 
    output_directory: '<dir>'

docker run -v /<dir>:/data gmd_0_9:demo /run_gmd.sh
```


## Getting started:

There are currently 4 parent folders in this repository that are intended to organize this repository:

1. Source: This folder contains the source code that runs the pipeline.
2. Examples: This folder contains example input configuration files, output, and performance metrics to help guide you when getting started. Note that this is intended as a starting place with no guarantee that these settings will work best for a given application. Parameters will need to be experimented with to best fit your application.
3. Output: This folder is the default location for all output from the GMD pipeline will go. The output will include the final csv files containing the compounds created within the pipeline, logs, and any error messages that may have come out of the pipeline
4. Scripts: This folder contains scripts that are intended to be ran either before or after using the pipeline. These scripts include data analys and visualizations scripts with varying purpose. See further documentation to learn how to best use those scripts

An example configuration can be found at 

`examples/LogP_JTVAE`

In this folder, there is a slurm file and configuration file that are intended to be used as an example case. Please edit the config file to point to your directory locations for each of the files. All necessary files can be found in the folder at FNLGMD/examples/LogP_JTVAE. To run the test case from the command line, please run the below command from the root folder of this repository:

`python source/main.py -config examples/LogP_JTVAE/config.yaml`

There is no "go to" values for many of the parameters. Each problem requires different settings and there is not quick way to determine the optimal values. Changing the parameters can have significant effects on the selection pressure, convergence, and diversity. It is important that you explore these parameters and learn about the effect each parameter has on the population as a whole. Please refer to the following README file that describes all of the required variables to be included in the config file:

`GGMD/examples/README.md`

The genetic algorithm (GA) is just a starting point. There are many modifications to canonical GA that can be done for particular applications. Consider trying new variations as well such as changing the selection strategy, introducing elitism into the population, or creating an adaptive GA where the mutation and crossover probabilities change depending on performance. This repository is intended to be modified and tailored to a specific use case. We will introduce some educational material into the repository in a later update to help guide new users in how to change parameters.

## Issues:

If you have any issues while using or installing this code, please create a new issue in the Issues tab of this github page. Someone will respond to the issue as soon as possible.

Feedback is always greatly appreciated! This code is still in development, changes will continue to be pushed and we have many plans for future development! 

## Development:

If you want to suggest changes to the code base, please create a fork from the master branch then create a merge request. Changes can include, but are not limited to implementing new generative models, scoring functions, optimization methods, and general bug fixes.
