import os
import yaml

import pandas as pd
from multiprocessing import freeze_support
from dask.distributed import Client, LocalCluster
import dask.array as da
import dask.dataframe as dd

# retrieve configuraion
with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)

DATA_DIR = config['data_dir']


if __name__ == '__main__':
    freeze_support()
    cluster = LocalCluster(n_workers=4)
    client = Client(cluster)
    print(client.dashboard_link)

    array = da.ones((10000, 1000, 1000), chunks=100)
    print(array.mean().compute())  # Should print 1.0

    df = dd.read_parquet(os.path.join(DATA_DIR,'benchmark', 'train.parquet'))
    print(df.shape, len(df))
    print(df.columns)
    print(df.head())

    county_size = df.groupby('county').size().compute()
    print(county_size)



    print('all done')