import time

import joblib
import os
import tempfile
import yaml
import sys
import gc
import numpy as np
import pickle
import psutil
import pandas as pd
from collections import Mapping, Container


# helper function to get approximate im-memory size of RF model in MB
# https://mljar.com/blog/random-forest-memory/
# DEPRECATED - Keeping only for references purposes
def rf_model_size_mb(model):
    raise DeprecationWarning('DEPRECATED')
    with tempfile.TemporaryDirectory() as tmpdir:
        rf_file = os.path.join(tmpdir, "rf_model")
        joblib.dump(model, rf_file, compress=0)
        rf_size = np.round(os.path.getsize(rf_file) / 1024 / 1024, 4)
        return rf_size


# another method for obtaining python object size
# https://towardsdatascience.com/the-strange-size-of-python-objects-in-memory-ce87bdfbb97f
def actualsize(input_obj):
    raise DeprecationWarning("DEPRECATED")
    memory_size = 0
    ids = set()
    objects = [input_obj]
    while objects:
        new = []
        for obj in objects:
            if id(obj) not in ids:
                ids.add(id(obj))
                memory_size += sys.getsizeof(obj)
                new.append(obj)
        objects = gc.get_referents(*new)
    return memory_size

def actualsize_mb(input_obj):
    return np.round(actualsize(input_obj) / (1024 * 1024), 4)


# Yet Another Memory Reporter
# https://code.tutsplus.com/tutorials/understand-how-much-memory-your-python-objects-use--cms-25609
# DOES REPORT CORRECTLY RF object
def deep_getsizeof(o, ids):
    """Find the memory footprint of a Python object

    This is a recursive function that drills down a Python object graph
    like a dictionary holding nested dictionaries with lists of lists
    and tuples and sets.

    The sys.getsizeof function does a shallow size of only. It counts each
    object inside a container as pointer only regardless of how big it
    really is.

    :param o: the object
    :param ids:
    :return:
    """
    raise DeprecationWarning('DEPRECATED')
    d = deep_getsizeof
    if id(o) in ids:
        return 0

    r = sys.getsizeof(o)
    ids.add(id(o))

    if isinstance(o, str) or isinstance(0, unicode):
        return r

    if isinstance(o, Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.iteritems())

    if isinstance(o, Container):
        return r + sum(d(x, ids) for x in o)

    return r

# get file size in MB
def get_file_size_mb(fp: str) -> np.float:
    return np.round(os.path.getsize(fp)/1024/1024, 4)


# load project configuration file
def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


    
class TestDataSource:
    def __init__(self):
        self.dataset_columns = {}
        self.combined_df = pd.DataFrame()
    
    @property
    def list_of_counties(self):
        return list(self.dataset_columns.keys())
    
    @property
    def shape(self):
        return self.combined_df.shape
    
    def add_dataset(self, fp):
        df = pd.read_csv(fp)
        county_id = os.path.basename(fp).split('.')[0]
        self.dataset_columns[county_id] = list(df.columns)
        df['county_id'] = county_id
        self.combined_df = pd.concat([self.combined_df,df])
        
        # pull out data set to make sure it matches the orginal
        df2 = self.extract_county_dataset(county_id)
        print(df.drop(['county_id'], axis=1).shape, df2.shape)
        assert df.drop(['county_id'], axis=1).shape == df2.shape
        assert np.all(df.drop(['county_id'], axis=1) == df2)
        
    def extract_county_dataset(self, county_id):
        return self.combined_df[self.combined_df['county_id'] == county_id][self.dataset_columns[county_id]].copy(deep=True)
    
    def scramble_dataset(self, random_seed = 123):
        self.combined_df = self.combined_df.sample(frac=1.0, random_state=random_seed)
        
    def get_record_by_row_number(self, row_number):
        row = self.combined_df.iloc[row_number]
        county_id = row.county_id
        row = pd.DataFrame(row).T
        return row[['county_id'] + self.dataset_columns[county_id]]