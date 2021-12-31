#!/bin/bash
#$-l rt_AF=1
#$-l h_rt=72:00:00 
#$-l USE_SSH=1
#$-v SSH_PORT=2245
#$-j y
#$-cwd
#$-o logs/bert_base_training_output.txt
#$-e logs/bert_base_training_error.txt

# source and load the modules
source /etc/profile.d/modules.sh
module load singularitypro/3.7

# run the job
singularity exec --nv --env OMP_NUM_THREADS=16 --bind $PWD:/workspace/bert \
    --bind $PWD/results:/results \
    ./karthik_bert_pytorch.sif \
    bash scripts/run_pretraining_base.sh
