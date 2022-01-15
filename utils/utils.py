
import os
import yaml
import numpy as np



# get file size in MB
def get_file_size_mb(fp: str) -> np.float:
    return np.round(os.path.getsize(fp)/1024/1024, 4)


# load project configuration file
def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


    
