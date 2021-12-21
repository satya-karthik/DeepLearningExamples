#!/bin/bash
#SBATCH --time=60
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --output=bert-training-%j.log

set -uxeC


# source and load the modules
source /etc/profile.d/modules.sh
module load singularity

export TEMP_TAG_NAME=karthik_bert_pytorch

SINGULARITY_NOHTTPS=true NO_PROXY=localhost singularity pull docker://localhost:5000/$TEMP_TAG_NAME

# run the job
singularity exec --nv --bind $PWD:/workspace/bert \
    --bind $PWD/results:/results \
    $TEMP_TAG_NAME \
    bash scripts/run_pretraining.sh