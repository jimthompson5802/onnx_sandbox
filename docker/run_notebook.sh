#!/bin/bash

# run jupyerlab notebook for
docker run -it --rm \
  -p 8888:8888 \
  -v $PWD:/home/jovyan/project \
  dsimages/onnx_testbed:v1 \
  jupyter lab --LabApp.token=''