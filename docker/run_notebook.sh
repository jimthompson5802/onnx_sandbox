#!/bin/bash

docker_repository=${2:-onnx_testbed}
tag=${1:-v1}

# run jupyerlab notebook for analysis work
# note: 2nd -v required for consistency of directory location for models and data
docker run -it --rm \
  -p 8888:8888 \
  -v $PWD:/home/jovyan/project \
  -v $PWD:/Users/jim/Desktop/onnx_sandbox \
  dsimages/${docker_repository}:${tag} \
  jupyter lab --LabApp.token=''