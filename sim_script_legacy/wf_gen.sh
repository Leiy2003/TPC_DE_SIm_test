#!/bin/bash

BATCH_SIZE=20
NUM_BATCHES=5


# for ((batch=0; batch<$NUM_BATCHES; batch++))
# do
#     for ((i=0; i<$BATCH_SIZE; i++))
#     do
#         param_index=$(($batch * $BATCH_SIZE + $i))
#         echo $param_index
#         python 3e_cut_wf_gen.py $param_index &
#         sleep 10
#     done
#     wait
# done

# for ((batch=0; batch<$NUM_BATCHES; batch++))
# do
#     for ((i=0; i<$BATCH_SIZE; i++))
#     do
#         param_index=$(($batch * $BATCH_SIZE + $i))
#         echo $param_index
#         python 4e_cut_wf_gen.py $param_index &
#         sleep 1
#     done
#     wait
# done

# for ((batch=0; batch<$NUM_BATCHES; batch++))
# do
#     for ((i=0; i<$BATCH_SIZE; i++))
#     do
#         param_index=$(($batch * $BATCH_SIZE + $i))
#         echo $param_index
#         python 5e_cut_wf_gen.py $param_index &
#         sleep 1
#     done
#     wait
# done

# for ((batch=0; batch<$NUM_BATCHES; batch++))
# do
#     for ((i=0; i<$BATCH_SIZE; i++))
#     do
#         param_index=$(($batch * $BATCH_SIZE + $i))
#         echo $param_index
#         python 6e_cut_wf_gen.py $param_index &
#         sleep 1
#     done
#     wait
# done

# for ((batch=0; batch<$NUM_BATCHES; batch++))
# do
#     for ((i=0; i<$BATCH_SIZE; i++))
#     do
#         param_index=$(($batch * $BATCH_SIZE + $i))
#         echo $param_index
#         python 7e_cut_wf_gen.py $param_index &
#         sleep 1
#     done
#     wait
# done

# for ((batch=0; batch<$NUM_BATCHES; batch++))
# do
#     for ((i=0; i<$BATCH_SIZE; i++))
#     do
#         param_index=$(($batch * $BATCH_SIZE + $i))
#         echo $param_index
#         python 8e_cut_wf_gen.py $param_index &
#         sleep 1
#     done
#     wait
# done

# for ((batch=0; batch<$NUM_BATCHES; batch++))
# do
#     for ((i=0; i<$BATCH_SIZE; i++))
#     do
#         param_index=$(($batch * $BATCH_SIZE + $i))
#         echo $param_index
#         python 9e_cut_wf_gen.py $param_index &
#         sleep 1
#     done
#     wait
# done

for ((batch=0; batch<$NUM_BATCHES; batch++))
do
    for ((i=0; i<$BATCH_SIZE; i++))
    do
        param_index=$(($batch * $BATCH_SIZE + $i))
        echo $param_index
        python 10e_cut_wf_gen.py $param_index &
        sleep 1
    done
    wait
done