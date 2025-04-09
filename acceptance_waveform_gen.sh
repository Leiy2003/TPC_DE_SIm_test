#!/bin/bash

BATCH_SIZE=10
NUM_BATCHES=5

for ((batch=0; batch<$NUM_BATCHES; batch++))
do
    for ((i=0; i<$BATCH_SIZE; i++))
    do
        param_index=$(($batch * $BATCH_SIZE + $i))
        echo $param_index
        python Acceptance_test.py $param_index &
        sleep 2
    done
    wait
done