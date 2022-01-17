conda activate onnx_training
python -m ipykernel install --user --name onnx_training
conda deactivate

conda activate onnx_scoring
python -m ipykernel install --user --name onnx_scoring
conda deactivate
