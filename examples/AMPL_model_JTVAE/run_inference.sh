#!/bin/bash

cd /data

#Activate AMPL's virtual environment
. /opt/venv/bin/activate

python /data/predict_from_model_file.py