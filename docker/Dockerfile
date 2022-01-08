FROM docker.io/jupyter/datascience-notebook:lab-3.2.5

COPY setup/onnx_sandbox.yaml /tmp/

# remove OSX specific build specification and OSX specific modules to allow for linux build
RUN  cat /tmp/onnx_sandbox.yaml | sed -e "s/=[^0-9].*$//"  | sed -e '/^prefix/d' \
    | sed -e '/appnope/d' \
    | sed -e '/libgfortran/d' | sed -e '/llvm\-openmp/d' >/tmp/conda_env.yaml \
    && mamba env create --file /tmp/conda_env.yaml && rm /tmp/conda_env.yaml

# TODO:  research if possible to define kernel during docker build.
# https://pythonspeed.com/articles/activate-conda-dockerfile/
# https://github.com/conda/conda/issues/2682
## make availble as kernel
#RUN conda run \
#    && source ~/.bashrc \
#    && conda activate onnx_sandbox && python -m ipykernel install --user --name onnx_sandbox