#!/usr/bin/env bash

# change this accordingly if you changed the name of image in the build script.
IMG_NAME=karthik/bert/pytorch

# function for running the image as non-root user
docker_run_user () {
    tempdir=$(mktemp -d)
    getent passwd > ${tempdir}/passwd
    getent group > ${tempdir}/group
    nvidia-docker run -v${HOME}:${HOME} -w$(pwd) --rm -u$(id -u):$(id -g\
    ) $(for i in $(id -G); do echo -n ' --group-add='$i; done) \
    -v ${tempdir}/passwd:/etc/passwd:ro \
    -v ${tempdir}/group:/etc/group:ro "$@"
}


# vs code or normal instance?
if [ "$1" == 'v' ]; then
    echo "launching vs code version..."
    CONTAINER_NAME=karthik_vs_pytorch_bert

elif [ "$1" == 'n' ]; then
    echo "launching normal version..."
    CONTAINER_NAME=karthik_pytorch_bert

else
    echo "invalid option, pass v or n as argument..."
    exit 1
fi



# you can configure ipc and net flags as per your requirements.
docker_run_user --name $CONTAINER_NAME -it --rm \
    --gpus device=all \
    --net=host \
    --ipc=host \
    -e LD_LIBRARY_PATH='/workspace/install/lib/' \
    -v "$PWD":/workspace/bert \
    -v "$PWD"/results:/results \
    "$IMG_NAME"
