#!/usr/bin/env bash

# if you change this, update launch script as well accordingly.
IMG_NAME=karthik/bert/pytorch

docker build --network=host --rm --pull --no-cache \
	-t $IMG_NAME -f ./scripts/docker/Dockerfile ./scripts/docker
