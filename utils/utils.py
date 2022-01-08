import joblib
import os
import tempfile
import yaml

import numpy as np

# helper function to get approximate im-memory size of RF model in MB
# https://mljar.com/blog/random-forest-memory/
def rf_model_size_mb(model):
    with tempfile.TemporaryDirectory() as tmpdir:
        rf_file = os.path.join(tmpdir, "rf_model")
        joblib.dump(model, rf_file, compress=0)
        rf_size = np.round(os.path.getsize(rf_file) / 1024 / 1024, 4)
        return rf_size

# load project configuration file
def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)