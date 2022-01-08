#!/bin/bash

# run jupyerlab notebook for analysis work
# note: 2nd -v required for consistency of directory location for models and data
docker run -it --rm \
  -p 8888:8888 \
  -v $PWD:/home/jovyan/project \
  -v $PWD:/Users/jim/Desktop/onnx_sandbox \
  dsimages/onnx_testbed:v1 \
  jupyter lab --LabApp.token=''