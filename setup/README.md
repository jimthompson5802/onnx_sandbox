# Software install on MacOS

Pre-requiste `conda`/`mamba` 

``` 
mamba env create --file onnx_sandbox.yaml
```

Note: `onnx_sandbox.yaml` is also used in building a docker image to run jupyter notebooks.  Some modifications are need to adapt the yaml specification from MacOS to Linux.  See `docker/Dockerfile` for details.