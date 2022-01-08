###
# Generates RF models and saves them as pickle and onnx files to be used
# in the scoring part of this benchmark.
###
import os
import shutil
import yaml
import pickle
from multiprocessing import freeze_support

import pandas as pd

import dask
from dask.distributed import LocalCluster, Client

from sklearn.ensemble import RandomForestRegressor
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


# retrieve configuration
with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)

DATA_DIR = config['data_dir']
MODELS_DIR = config['models_dir']

# set up models directory
models_test_dir = os.path.join(MODELS_DIR, 'benchmark')
shutil.rmtree(models_test_dir, ignore_errors=True)
os.makedirs(models_test_dir)

# train one county model and save as pickle and onnx files
def train_a_model(county_id: str) -> None:
    print(f'Training county: {county_id}')
    train_all_df = pd.read_parquet(os.path.join(DATA_DIR, 'benchmark', 'train.parquet'))
    train_county = train_all_df.loc[train_all_df['county'] == county_id]

    X = train_county.drop(['county', 'y'], axis='columns')
    y = train_county['y']

    rf = RandomForestRegressor(n_jobs=4, random_state=config['random_seed'])
    rf.fit(X, y)
    # print(f'RF fit() complete for {county_id}')

    # save model as pickle file
    with open(os.path.join(models_test_dir, county_id+'.pkl'), 'wb') as f:
        pickle.dump(rf, f)
    # print(f"complete rf sklearn save for {county_id}")

    # save model as onnx
    explanatory_var = [('float_input', FloatTensorType([None, 20]))]
    onnx_model = convert_sklearn(rf, initial_types=explanatory_var)
    with open(os.path.join(models_test_dir, county_id+'.onnx'), 'wb') as f:
        f.write(onnx_model.SerializeToString())
    # print(f'complete rf onnx save for {county_id}')

    print(f"completed training for {county_id}")

    return {county_id: "OK"}


if __name__ == '__main__':
    freeze_support()

    # get list of counties from training data set
    df = pd.read_parquet(os.path.join(DATA_DIR, 'benchmark', 'train.parquet'))
    county_list = df['county'].unique().tolist()
    del df

    # setup dask cluster
    cluster = LocalCluster(n_workers=3)
    client = Client(cluster)
    print(client.dashboard_link)

    # kick off training
    future_status = []
    for county_id in county_list:
        status = client.submit(train_a_model, county_id)
        future_status.append(status)

    # wait for all training to complete
    results = client.gather(future_status)

    print(results)

    client.close()
    cluster.close()




