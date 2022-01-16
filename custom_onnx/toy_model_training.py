#!/usr/bin/env python
# coding: utf-8

# # Setup toy sklearn custom image
# 
# Simple linear model $y = 2*x1 + 4*x2 + 1$

# In[1]:


from sklearn.base import BaseEstimator, RegressorMixin

import numpy as np
import pandas as pd
from io import StringIO

from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import update_registered_converter
from skl2onnx.common.data_types import guess_numpy_type
from skl2onnx.algebra.onnx_ops import OnnxAdd, OnnxMul, OnnxReduceSum

import onnxruntime as rt


# ## Custom Class Defintion

class CustomRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        # assume following are model weights determined by "fit()" method
        self.coeff_ = np.array([2, 4], dtype=np.float32)
        
    def predict(self, X):
        part1 = self.coeff_ * X
        return  np.sum(part1, axis=1) + 1


custom_regressor = CustomRegressor()


# ## Create some test data

test_data = """x1,x2,y
1,2,11
-3,1,-1
0,0,1
1,1,7
"""
test_df = pd.read_csv(StringIO(test_data))
print(f'test data:\n{test_df}')


# check predictions
print(f"predictions: {custom_regressor.predict(test_df.drop(['y'], axis=1).to_numpy())}")
print(f"R2 score: {custom_regressor.score(test_df.drop(['y'], axis=1).to_numpy(), test_df['y'])}")


# ## Save in onnx format

# ### Helper Functions
# shape calculator
def custom_regressor_shape_calculator(operator):
    op = operator.raw_operator
    input_type = operator.inputs[0].type.__class__
    # The shape may be unknown. *get_first_dimension*
    # returns the appropriate value, None in most cases
    # meaning the transformer can process any batch of observations.
    input_dim = operator.inputs[0].get_first_dimension()
    output_type = input_type([input_dim, 1])
    operator.outputs[0].type = output_type

def custom_regressor_converter(scope, operator, container):
    op = operator.raw_operator
    opv = container.target_opset
    out = operator.outputs

    # We retrieve the unique input.
    X = operator.inputs[0]

    # In most case, computation happen in floats.
    # But it might be with double. ONNX is very strict
    # about types, every constant should have the same
    # type as the input.
    dtype = guess_numpy_type(X.type)

    # We tell in ONNX language how to compute the unique output.
    # Y = np.sum(coeff_ * X, axis=1) + 1
    # op_version=opv tells which opset is requested

    # element-wise multiplication, op.coeff_ broadcasted
    part1 = OnnxMul(X, op.coeff_.astype(dtype), op_version=opv)

    # sum along axis=1, the np.array() designates axis to reduce
    sum_part1 = OnnxReduceSum(part1, np.array([1]), op_version=opv )

    # add constant 1 to each result for final prediction, np.array() is the constant "1"
    Y = OnnxAdd(sum_part1, np.array([1]).astype(dtype), op_version=opv,
                output_names=out[:1])

    # add to computational graph
    Y.add_to(scope, container)

# register custom model with onnx
update_registered_converter(
    CustomRegressor, "MyCustomRegressor",
    custom_regressor_shape_calculator,
    custom_regressor_converter)

# save model as onnx
# test_df.drop(['y'], axis=1).astype(np.float32)
explanatory_var = [('float_input', FloatTensorType([None, 2]))]
onnx_model = convert_sklearn(custom_regressor, initial_types=explanatory_var)

with open('./toy_model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

# restore model and run predictions
sess = rt.InferenceSession('./toy_model.onnx')

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
prediction = sess.run([label_name], {'float_input': test_df.drop(['y'], axis=1).to_numpy().astype(np.float32)})[0]
print(f'prediction: {prediction.shape}\n {prediction} ')
prediction = prediction.reshape(-1,)
print(prediction.shape)
print('restored model', np.max(np.abs(test_df['y'] - prediction)))

print('all done')




