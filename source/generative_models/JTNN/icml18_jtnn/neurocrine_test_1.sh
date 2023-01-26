#!/bin/bash
# Test recovery rate and average Tanimoto distance for SMILES strings encoded and decoded by
# trained JT-VAEs

#SBATCH -A ncov2019
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH --export=ALL
#SBATCH -D /p/vast1/kmelough/neurocrine_vae/slurm
start=`date +%s`

python -m atomsci.glo.generative_networks.icml18_jtnn.test_encoder --mode hyperparam \
  --vae_path /p/vast1/kmelough/neurocrine_vae/models \
  --test_path /g/g14/kmelough/git/glo/atomsci/glo/generative_networks/icml18_jtnn/data/neurocrine/neurocrine_test_smiles.txt \
  --train_path /g/g14/kmelough/git/glo/atomsci/glo/generative_networks/icml18_jtnn/data/neurocrine/neurocrine_train_smiles.txt \
  --max_jobs 80 \
  --max_size 1000 \
  --bank ncov2019 \
  --partition pbatch --time_limit 1440 \
  --show_recovery \
  --verbose
