import joblib
import os
import tempfile
import yaml
import sys
import gc
import numpy as np

# helper function to get approximate im-memory size of RF model in MB
# https://mljar.com/blog/random-forest-memory/
def rf_model_size_mb(model):
    with tempfile.TemporaryDirectory() as tmpdir:
        rf_file = os.path.join(tmpdir, "rf_model")
        joblib.dump(model, rf_file, compress=0)
        rf_size = np.round(os.path.getsize(rf_file) / 1024 / 1024, 4)
        return rf_size


# another method for obtaining python object size
# https://towardsdatascience.com/the-strange-size-of-python-objects-in-memory-ce87bdfbb97f
def actualsize_mb(input_obj):
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
    return np.round(memory_size / 1024 / 1024, 4)

# load project configuration file
def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)