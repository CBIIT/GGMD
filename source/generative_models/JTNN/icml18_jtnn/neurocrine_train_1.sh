#!/bin/bash

#SBATCH -A ncov2019
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH --export=ALL
#SBATCH -D /p/vast1/kmelough/neurocrine_vae/slurm
start=`date +%s`

export DATA_DIR=/g/g14/kmelough/git/glo/atomsci/glo/generative_networks/icml18_jtnn/data/neurocrine

cd /g/g14/kmelough/git/glo/atomsci/glo/generative_networks/icml18_jtnn

python -m atomsci.glo.generative_networks.icml18_jtnn.jtnn_hyperparam \
 --n_models 80 \
 --workers 36 \
 --save_dir /p/vast1/kmelough/neurocrine_vae/models \
 --save_interval 1 \
 --train_split data/neurocrine/processed \
 --vocab data/neurocrine/vocab.txt \
 --time_limit 1440 \
 --partition pbatch
