#!/bin/bash

# Assign command line arguments to variables
initial_pop_size=$1
max_population=$2

source activate gmd
cd /FNLGMD/examples/LogP_JTVAE
rm all_vocab.txt # remove this file from container for testing
mkdir -p /mnt/output

# Update paths
sed -i "s|smiles_input_file: './examples/LogP_AutoGrow/zinc_smiles.txt'|smiles_input_file: 'all.txt'|g" config.yaml
sed -i "s|output_directory: './examples/LogP_JTVAE/'|output_directory: '/mnt/output'|g" config.yaml
sed -i "s|vocab_path: './examples/LogP_JTVAE/all_vocab.txt'|vocab_path: '/mnt/all_vocab.txt'|g" config.yaml
sed -i "s|model_path: './examples/LogP_JTVAE/model.epoch-35'|model_path: 'model.epoch-35'|g" config.yaml

# Update initial_pop_size and max_population
sed -i "s|initial_pop_size: [0-9]*|initial_pop_size: $initial_pop_size|g" config.yaml
sed -i "s|max_population: [0-9]*|max_population: $max_population|g" config.yaml

python /FNLGMD/source/main.py -config config.yaml

