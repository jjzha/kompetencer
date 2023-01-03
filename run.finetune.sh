#!/bin/bash

# we called our virtualenv 'komp'
source ./komp/bin/activate

MODEL=$1
PARAMETERS=$2

for c in 1 2 3 4 5
do

mkdir logs
echo "Training $MODEL on $PARAMETERS"
python3 train.py --dataset_config configs/$PARAMETERS.json --parameters_config configs/$MODEL/$MODEL.$c.json --device 0 --name logs/esco.new.$MODEL.$PARAMETERS.$c

done

deactivate
