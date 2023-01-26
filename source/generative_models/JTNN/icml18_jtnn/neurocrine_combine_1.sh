#!/bin/bash
# Combine recovery performance outputs from multiple models in a hyperparameter run.


python -m atomsci.glo.generative_networks.icml18_jtnn.test_encoder --mode combine \
  --vae_path /p/vast1/kmelough/neurocrine_vae/models \
  --output_file /p/vast1/kmelough/neurocrine_vae/combined_recovery_perf.csv
