#!/bin/bash
#$-l rt_AF=1
#$-l h_rt=72:00:00 
#$-l USE_SSH=1
#$-v SSH_PORT=2245
#$-j y
#$-cwd
#$-o logs/bert_large_training_output.txt
#$-e logs/bert_large_training_error.txt



##################################### slurm settings ######################################## 
# uncomment the below line for debug
# set -uxeC
# use the below line for normal usage. 
# note: don't use both.
set -ueC

########## copy files to the local storage ##########

#### setting the directories ####

# these are the profile and group storage (permanent storage).
LOCAL_CODE="$PWD/"
LOCAL_DATA="$HOME/gcb50379/dataset/BERT_pre_training/data/"
LOCAL_RESULTS="$HOME/gcb50379/dataset/BERT_pre_training/results/"

# these are temporary storage provided on the GPU instance.
TEMP_CODE="$SGE_LOCALDIR/code/"
TEMP_DATA="$SGE_LOCALDIR/data/"

# copy code and data dirs into temp storage
echo "copying from $LOCAL_CODE to $TEMP_CODE"
cp -r "$LOCAL_CODE" "$TEMP_CODE"

if [ $? -eq 0 ]; then
    echo "code has been copied to TEMP storage: $TEMP_CODE"
else
    echo "something went wrong while coping the code"
fi

echo "copying from $LOCAL_DATA to $TEMP_DATA"
cp -r "$LOCAL_DATA" "$TEMP_DATA"

if [ $? -eq 0 ]; then
    echo "data has been copied to TEMP storage: $TEMP_DATA"
else
    echo "something went wrong while coping the data"
fi


# source and load the modules
source /etc/profile.d/modules.sh
module load singularitypro/3.7

########## running the job ##########

# singularity will have 3 mappings
# /workspace/bert ==> TEMP_CODE
# /data ==> TEMP_DATA
# /results ==> LOCAL_RESULTS

singularity exec --nv --env OMP_NUM_THREADS=16 \
    --bind "$TEMP_CODE":/workspace/bert \
    --bind "$TEMP_DATA":/data \
    --bind "$LOCAL_RESULTS":/results \
    ./karthik_bert.simg \
    bash scripts/run_pretraining_large.sh
