#!/bin/bash

WRK="/GGMD/source"
echo $WRK
cd $WRK

CODE="main.py"
CONF="/data/config.yaml"

source activate gmd

python $CODE -config $CONF

echo "Done"