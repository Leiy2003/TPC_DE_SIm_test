#!/bin/bash

BATCH_SIZE=1
NUM_BATCHES=1

for ((batch=0; batch<$NUM_BATCHES; batch++))
do
    for ((i=0; i<$BATCH_SIZE; i++))
    do
        param_index=$(($batch * $BATCH_SIZE + $i))
        echo $param_index
        python 8e_cut_wf_gen.py $param_index &
        sleep 10
    done
    wait
done