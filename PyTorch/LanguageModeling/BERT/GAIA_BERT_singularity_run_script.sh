#!/bin/bash
#SBATCH --time=1440
#SBATCH --ntasks=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --output=bert-training-%j.log


##################################### slurm settings ######################################## 
# uncomment the below line for debug
# set -uxeC
# use the below line for normal usage. 
# note: don't use both.
set -ueC

# source and load the modules
source /etc/profile.d/modules.sh
module load singularity/3.5.3

#################################  setting the directories ################################## 

# these are the profile and group storage (permanent storage).
IMAGE_PATH=$HOME/images/karthik_bert.simg
LOCAL_CODE="$PWD/"
LOCAL_DATA="$HOME/Datasets/bert-pre-training/data/"
LOCAL_RESULTS="$HOME/Datasets/bert-pre-training/results/"

# these are temporary storage provided on the GPU instance.
LOCALDIR="/local/job/$SLURM_JOB_ID"
TEMP_CODE="$LOCALDIR/code/"
TEMP_DATA="$LOCALDIR/data/"

############################## copy files to the local storage ##############################
SECONDS=0
echo "start copying"
echo "copying from $LOCAL_CODE to $TEMP_CODE on $SLURM_JOB_NUM_NODES nodes"
srun --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 cp -r "$LOCAL_CODE" "$TEMP_CODE"

if [ $? -eq 0 ]; then
    echo "code has been copied to TEMP storage: $TEMP_CODE"
else
    echo "something went wrong while coping the code"
fi

echo "copying from $LOCAL_DATA to $TEMP_DATA on $SLURM_JOB_NUM_NODES nodes"
srun --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 cp -r "$LOCAL_DATA" "$TEMP_DATA"

if [ $? -eq 0 ]; then
    echo "data has been copied to TEMP storage: $TEMP_DATA"
else
    echo "something went wrong while coping the data"
fi
echo "copying finished/aborted in $SECONDS secs"
############################## done copying files to the local storage #######################

###################################### running the job #######################################

# singularity will have 3 mappings
# /workspace/bert ==> TEMP_CODE
# /data ==> TEMP_DATA
# /results ==> LOCAL_RESULTS

singularity exec --nv --env OMP_NUM_THREADS=16 \
    --bind "$TEMP_CODE":/workspace/bert \
    --bind "$TEMP_DATA":/data \
    --bind "$LOCAL_RESULTS":/results \
    $IMAGE_PATH \
    bash scripts/run_pretraining_base.sh



# source and load the modules
source /etc/profile.d/modules.sh
module load singularity

TEMP_TAG_NAME=karthik_bert_pytorch

# run the job
singularity exec --nv --bind $PWD:/workspace/bert \
    --bind $PWD/results:/results \
    $TEMP_TAG_NAME \
    bash scripts/run_pretraining.sh
