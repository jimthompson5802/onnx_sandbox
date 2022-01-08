#!/bin/bash

image_tag=${1:-v1}
image_repository=${2:-onnx_testbed}
docker_file=${3:-Dockerfile}
progress=${4:-auto}

docker build --progress=${progress} \
  -t dsimages/${image_repository}:${image_tag} \
  -f ./${docker_file}  ..
