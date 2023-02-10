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

