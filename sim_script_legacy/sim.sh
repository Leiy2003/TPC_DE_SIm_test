#!/bin/bash

BATCH_SIZE=50
NUM_BATCHES=400

for ((batch=0; batch<$NUM_BATCHES; batch++))
do
    for ((i=0; i<$BATCH_SIZE; i++))
    do
        param_index=$(($batch * $BATCH_SIZE + $i))
        echo $param_index
        python sim.py $param_index DE_param_30_test.yaml&
        sleep 2
    done
    wait
done