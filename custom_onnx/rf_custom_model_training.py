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
from skl2onnx.algebra.onnx_ops import OnnxTreeEnsembleRegressor
from skl2onnx.shape_calculators.ensemble_shapes import calculate_tree_regressor_output_shapes
from skl2onnx.operator_converters.random_forest import convert_sklearn_random_forest_regressor_converter

import onnxruntime as rt

# define custom RF model class
class CustomRandomForestRegressor(RandomForestRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # override standard predict method
    # modifications:
    #    invoke super class predict() - super_prediction
    #    invoke individual tree estimators
    #    compute mean of collection of tree estimators - tree_mean_prediction
    #    return super_prediction - tree_mean_prediction, this should result in all zero predictions
    def predict(self, X):
        # obtain standard RF prediction
        super_prediction = super().predict(X)

        # compute predictions directly from trees
        tree_predictions = [e.predict(X.to_numpy()) for e in self.estimators_]
        tree_mean_prediction = np.mean(np.array(tree_predictions), axis=0)

        # return the difference between the two predictions, values should all be zero
        return super_prediction - tree_mean_prediction

# setup custom RF model
rf = CustomRandomForestRegressor(
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


# # save model as onnx
# ### Helper Functions
# shape calculator
def custom_regressor_shape_calculator(operator):
    # TODO: may not need updating for custom predict() that returns zeros
    # # op = operator.raw_operator
    # input_type = operator.inputs[0].type.__class__
    # # The shape may be unknown. *get_first_dimension*
    # # returns the appropriate value, None in most cases
    # # meaning the transformer can process any batch of observations.
    # input_dim = operator.inputs[0].get_first_dimension()
    # output_type = input_type([input_dim, 1])
    # operator.outputs[0].type = output_type
    # operator.outputs[0].type.shape = [input_dim, 1]

    # call standard sklearn shape calculator
    calculate_tree_regressor_output_shapes(operator)

def custom_regressor_converter(scope, operator, container):
    # TODO: need to figure out the custom ops to return zero predictions
    # # op = operator.raw_operator
    # opv = container.target_opset
    # out = operator.outputs
    #
    # # We retrieve the unique input.
    # X = operator.inputs[0]
    #
    # # In most case, computation happen in floats.
    # # But it might be with double. ONNX is very strict
    # # about types, every constant should have the same
    # # type as the input.
    # dtype = guess_numpy_type(X.type)
    #
    # # We tell in ONNX language how to compute the unique output.
    # # Y = OnnxTreeEnsembleRegressor
    # # op_version=opv tells which opset is requested
    #
    # # add constant 1 to each result for final prediction, np.array() is the constant "1"
    # Y = OnnxTreeEnsembleRegressor(X, op_version=opv,
    #             output_names=out[:1])
    #
    # # add to computational graph
    # Y.add_to(scope, container)

    # TODO: we may need to provide custom equivalent
    # call standard sklearn converter
    convert_sklearn_random_forest_regressor_converter(scope, operator, container)


# register custom model with onnx
update_registered_converter(
    CustomRandomForestRegressor,
    "CustomRandomForestRegressor",
    custom_regressor_shape_calculator,  # custom built
    custom_regressor_converter,  # custom built
    # required to support standard sklearn RF class
    options={
        'decision_path': [True, False],
        'decision_leaf': [True, False]
    }
)


explanatory_var = [('float_input', FloatTensorType([None, 2]))]
onnx_model = convert_sklearn(rf, initial_types=explanatory_var)

with open('./rf_custom_model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

# restore model and run predictions
sess = rt.InferenceSession('./rf_custom_model.onnx')

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
prediction = sess.run([label_name], {'float_input': test_df.drop(['y'], axis=1).to_numpy().astype(np.float32)})[0]
print(f'\nonnx model prediction: {prediction.shape}\n {prediction} ')
prediction = prediction.reshape(-1,)
print(prediction.shape)
print('restored onnx model', np.max(np.abs(test_df['y'] - prediction)))

# run predictions again
prediction = sess.run([label_name], {'float_input': test_df.drop(['y'], axis=1).to_numpy().astype(np.float32)})[0]
print(f'\n2nd onnx model prediction: {prediction.shape}\n {prediction} ')

print('all done')




