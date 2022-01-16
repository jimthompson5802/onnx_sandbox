import os
import shutil
import yaml
import pickle
import sys

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from skl2onnx import to_onnx, convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


# retrieve configuration
with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)

DATA_DIR = config['data_dir']
MODELS_DIR = config['models_dir']

# retrieve one county data to train
county_id = 'cnty0000'
train_all_df = pd.read_parquet(os.path.join(DATA_DIR, 'train.parquet'))
train_county = train_all_df.loc[train_all_df['county'] == county_id]

X = train_county.drop(['county', 'y'], axis='columns')
y = train_county['y']

rf = RandomForestRegressor(
    n_estimators=8,
    max_depth=4,
    random_state=config['random_seed']
)
rf.fit(X, y)

# save model as pickle file
with open(os.path.join(MODELS_DIR, county_id+'.pkl'), 'wb') as f:
    pickle.dump(rf, f)

# save model as onnx
explanatory_var = [('float_input', FloatTensorType([None, 20]))]
onnx_model = convert_sklearn(rf, initial_types=explanatory_var)
print(onnx_model)
with open(os.path.join(MODELS_DIR, county_id+'.onnx'), 'wb') as f:
    f.write(onnx_model.SerializeToString())

print('all done')




