###
# Generates RF models and saves them as pickle and onnx files to be used
# in the scoring part of this benchmark.
###
import argparse
import os
import shutil
import yaml
import pickle
from typing import Dict

import pandas as pd

from dask.distributed import LocalCluster, Client

from sklearn.ensemble import RandomForestRegressor
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# this dummy import needed for work-around re: this issue
# https://github.com/dask/distributed/issues/4168
# apparently there is a Python 3.9 and dask incompatibility
import multiprocessing.dummy


# get command line arguments
parser = argparse.ArgumentParser(description='Train RF models for testing')
parser.add_argument('--config', dest='config_file', action='store',
                    required=True,
                    help='benchmark configuration yaml file')

args = parser.parse_args()

# retrieve configuration
with open(args.config_file, 'r') as f:
    config = yaml.safe_load(f)

DATA_DIR = config['data_dir']
MODELS_DIR = config['models_dir']

# set up models directory
shutil.rmtree(MODELS_DIR, ignore_errors=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# train one county model and save as pickle and onnx files
def train_a_model(county_id: str) -> Dict:
    print(f'Training county: {county_id}')
    try:
        train_df = pd.read_parquet(os.path.join(DATA_DIR, 'train.parquet'))
        train_df = train_df.loc[train_df['county'] == county_id]

        X = train_df.drop(['county', 'y'], axis='columns')
        y = train_df['y']

        rf = RandomForestRegressor(
            n_estimators=config['number_trees'],
            n_jobs=2,
            random_state=config['random_seed']
        )
        rf.fit(X, y)
        # print(f'RF fit() complete for {county_id}')

        # save model as pickle file
        with open(os.path.join(MODELS_DIR, county_id+'.pkl'), 'wb') as f:
            pickle.dump(rf, f)
        # print(f"complete rf sklearn save for {county_id}")

        # save model as onnx
        explanatory_var = [('float_input', FloatTensorType([None, 20]))]
        onnx_model = convert_sklearn(rf, initial_types=explanatory_var)
        with open(os.path.join(MODELS_DIR, county_id+'.onnx'), 'wb') as f:
            f.write(onnx_model.SerializeToString())
        # print(f'complete rf onnx save for {county_id}')

        print(f"completed training for {county_id}")

        return {county_id: "OK"}
    except Exception as e:
        return {county_id: f'FAILED: {e}'}


if __name__ == '__main__':

    # get list of counties from training data set
    df = pd.read_parquet(os.path.join(DATA_DIR, 'train.parquet'))
    county_list = df['county'].unique().tolist()
    del df

    # setup dask cluster
    cluster = LocalCluster(n_workers=6)
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

    # shutdown the clusters
    client.close()
    cluster.close()




