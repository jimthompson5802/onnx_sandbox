#!/usr/bin/env python
# coding: utf-8


from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor

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


rf = RandomForestRegressor(
    n_estimators=8,
    random_state=13
)


# ## Create some test data

test_data = """x1,x2,y
1,2,11
-3,1,-1
0,0,1
1,1,7
"""
test_df = pd.read_csv(StringIO(test_data))
print(f'test data:\n{test_df}')

# train rf model
rf.fit(test_df.drop(['y'], axis=1), test_df['y'])


# check predictions
rf_prediction = rf.predict(test_df.drop(['y'], axis=1))
print(f"predictions: {rf_prediction}")
print(f"R2 score: {rf.score(test_df.drop(['y'], axis=1), test_df['y'])}")
print('sklearn model', np.max(np.abs(test_df['y'] - rf_prediction)))


# save model as onnx
# test_df.drop(['y'], axis=1).astype(np.float32)
explanatory_var = [('float_input', FloatTensorType([None, 2]))]
onnx_model = convert_sklearn(rf, initial_types=explanatory_var)

with open('./rf_standard_model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

# restore model and run predictions
sess = rt.InferenceSession('./rf_standard_model.onnx')

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
prediction = sess.run([label_name], {'float_input': test_df.drop(['y'], axis=1).to_numpy().astype(np.float32)})[0]
print(f'prediction: {prediction.shape}\n {prediction} ')
prediction = prediction.reshape(-1,)
print(prediction.shape)
print('restored onnx model', np.max(np.abs(test_df['y'] - prediction)))

print('all done')




