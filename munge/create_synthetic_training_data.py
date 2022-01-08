import os
import shutil
import yaml

import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


#  get configuration
with open('../config.yaml', 'r') as f:
    config  = yaml.safe_load(f)

DATA_DIR = config['data_dir']
NUMBER_RECORDS = config['number_records']
NUMBER_COUNTIES = config['number_counties']
NUMBER_FEATURES = config['number_features']
NUMBER_INFORMATIVE = config['number_informative']
RANDOM_SEED = config['random_seed']
FRACTION_FOR_TEST = config['fraction_for_test']

# generate county identifier
cnty_id = np.array(
    [f'cnty{n:04}' for n in np.random.choice(NUMBER_COUNTIES, size=NUMBER_RECORDS, replace=True)]
).reshape(-1, 1)

# generate raw explanatory and response variables.
X, y = make_regression(NUMBER_RECORDS, NUMBER_FEATURES,
                n_informative=NUMBER_INFORMATIVE,
                random_state=RANDOM_SEED)


# combine into a data frame
df = pd.DataFrame(np.hstack([cnty_id, X, y.reshape(-1, 1)]))
df.columns = ['county'] + [f'X_{n:02}' for n in range(NUMBER_FEATURES)] + ['y']

# setup up benchmark data directory
benchmark_data_dir = os.path.join(DATA_DIR, 'benchmark')
shutil.rmtree(benchmark_data_dir, ignore_errors=True)
os.makedirs(benchmark_data_dir, exist_ok=True)

# save sample of data as csv
df.sample(10, random_state=RANDOM_SEED)\
    .to_csv(os.path.join(benchmark_data_dir, "sample.csv"), index=False)

# create train/test split
train_df, test_df = train_test_split(df, test_size=FRACTION_FOR_TEST)

# save training and test data set
train_df.to_parquet(os.path.join(benchmark_data_dir, 'train.parquet'), index=False)
test_df.to_parquet(os.path.join(benchmark_data_dir, 'test.parquet'), index=False)


print('all done')