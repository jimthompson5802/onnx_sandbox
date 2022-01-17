import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import onnx
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
import onnxruntime as ort

from skl2onnx.algebra.onnx_ops import OnnxTranspose, OnnxAdd, OnnxMul
from skl2onnx.common.data_types import FloatTensorType

# add to matrices
node1 = OnnxAdd('X','Y', output_names=['Sum1'])  # output_names is opttional

# get negative of the sum
node2 = OnnxMul(node1, np.array([-1]).astype(np.float32))

# add negative sum with original sum, should be all zeros
node3 = OnnxAdd(node1, node2, output_names=['Z'])

# Create place holders for the input variables
X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [None, 2])
Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [None, 2])

# create model from terminal node
onnx_model3 = node3.to_onnx({'X': X, 'Y': Y})

# Check the model, model ok if no messages are generated
onnx.checker.check_model(onnx_model3)

X_in = np.arange(6).reshape(-1,2).astype(np.float32)
Y_in = 10*X_in
print(f'X=\n{X_in},\nY=\n{Y_in}')
print(f'sum=\n{X_in + Y_in}')

sess = ort.InferenceSession(onnx_model3.SerializeToString())
names = [i.name for i in sess.get_inputs()]
dinputs = {name: input for name, input in zip(names, [X_in, Y_in])}
res = sess.run(None, dinputs)
names = [o.name for o in sess.get_outputs()]
prediction = {name: output for name, output in zip(names, res)}

print(f"predictions:\n{prediction['Z']}")

print(f"prediction.shape: {prediction['Z'].shape}")