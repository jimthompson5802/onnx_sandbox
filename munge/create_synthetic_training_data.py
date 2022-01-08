import os

import pandas as pd
import numpy as np
from sklearn.datasets import make_regression


DATA_DIR = '../data'
NUMBER_RECORDS = 100000
NUMBER_COUNTIES = 20
NUMBER_FEATURES = 20
NUMBER_INFORMATIVE = 14
RANDOM_SEED = 123

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

# save to storage
training_dir = os.path.join(DATA_DIR, 'training')
os.makedirs(training_dir, exist_ok=True)

# save sample of data as csv
df.sample(10, random_state=RANDOM_SEED)\
    .to_csv(os.path.join(training_dir, "train_sample.csv"), index=False)

# save full data set
df.to_parquet(os.path.join(training_dir, 'training.parquet'), index=False)

print('all done')