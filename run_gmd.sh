#!/bin/bash

source /ext3/env.sh
conda info --envs

WRK="/FNLGMD/source"
echo $WRK
cd $WRK

CODE="/source/main.py"
CONF="/data/config.yaml"

python $CODE -config $CONF

echo "Done"