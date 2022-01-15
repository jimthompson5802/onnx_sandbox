# Software install on MacOS

Pre-requiste `conda`/`mamba` 

``` 
mamba env create --file onnx_training.yaml
mamba env create --file onnx_scoring.yaml
```

Following build conda environments to support training sklearn model and saving in onnx format and to run onnx inference.  These are also used by Docker build to create jupyterlab images.

* `onnx_training.yaml` conda environment to train and create onnx model.
* `onnx_scoring.yaml` conda environment to do onnx inferencing (Does not contain any sklearn packages.)