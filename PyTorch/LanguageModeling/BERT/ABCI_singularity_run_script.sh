#!/bin/bash
#$-l rt_AF=1
#$-l h_rt=1:00:00 
#$-l USE_SSH=1
#$-v SSH_PORT=2245
#$-j y
#$-cwd

# source and load the modules
source /etc/profile.d/modules.sh
module load cuda/11.0/11.0.3 singularitypro/3.7 \
    cudnn/8.0/8.0.5 python/3.6/3.6.12

# run the job
singularity exec --nv --bind $PWD:/workspace/bert \
    --bind $PWD/results:/results \
    ./karthik_bert_pytorch.sif \
    bash scripts/run_pretraining.sh