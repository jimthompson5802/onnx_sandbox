import time

import joblib
import os
import tempfile
import yaml
import sys
import gc
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor
from onnxruntime import InferenceSession

import onnxruntime as rt

SKLEARN = 'sklearn'
ONNX = 'onnx'

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

# get file size in MB
def get_file_size_mb(fp: str) -> np.float:
    return np.round(os.path.getsize(fp)/1024/1024, 4)


# load project configuration file
def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


class BenchmarkDriver:
    def __init__(
            self,
            model_type: str,
            models_dir: str,
            performance_fp: str,
            test_scenario: str
    ) -> None:
        self.model_type = model_type
        if model_type not in {SKLEARN, ONNX}:
            raise ValueError(
                f'model_type={model_type} invalid, should be "sklearn" or "onnx".'
            )
        self.models_dir = models_dir
        self.performance_file = open(performance_fp, 'wt')
        self.performance_file.write(
            "scenario,record_id,model_type,county_id,model_memory_size_mb,"
            f"model_load_time,model_score_time,prediction\n"
        )
        self.test_scenario = test_scenario

        # performance metrics
        self.model_load_time = 0.0
        self.model_score_time = 0.0
        self.predicted_score = np.nan
        self.county_id = ''
        self.model_memory_size_mb = 0
        self.record_id = -1

    def clear_stats(self):
        self.model_load_time = 0.0
        self.model_score_time = 0.0
        self.predicted_score = np.nan
        self.county_id = ''
        self.model_memory_size_mb = 0
        self.record_id = -1

    def write_performance_data(self):
        self.performance_file.write(
            f'{self.test_scenario},{self.record_id},{self.model_type},{self.county_id},{self.model_memory_size_mb},'
            f'{self.model_load_time},{self.model_score_time},{self.predicted_score}\n'
        )

    def close_performance_data(self):
        self.performance_file.close()

    def score_one_record(self, county_id: str, record_id: int, record: np.array):
        self.county_id = county_id
        self.record_id = record_id
        if self.model_type == SKLEARN:
            model: RandomForestRegressor = self._retrieve_sklearn_model(county_id)
            self.predicted_score = self._predict_sklearn_model(model, record)
        else:
            model: InferenceSession = self._retrieve_onnx_model(county_id)
            self.predicted_score = self._predict_onnx_model(model, record)

        # record performance
        self.write_performance_data()
        self.clear_stats()


    def _retrieve_sklearn_model(self, county_id: str) -> RandomForestRegressor:
        # retrieve model from persistent storage
        t0 = time.perf_counter()
        with open(os.path.join(self.models_dir, county_id + '.pkl'), 'rb') as f:
            rf_pkl_model = pickle.load(f)
        self.model_load_time = time.perf_counter() - t0
        self.model_memory_size_mb = actualsize_mb(rf_pkl_model)
        return rf_pkl_model

    def _retrieve_onnx_model(self, county_id: str) -> InferenceSession:
        # retrieve model from file
        t0 = time.perf_counter()
        sess = rt.InferenceSession(os.path.join(self.models_dir, county_id + '.onnx'))
        self.model_load_time = time.perf_counter() - t0
        self.model_memory_size_mb = actualsize_mb(sess)
        return sess

    def _predict_sklearn_model(self, model, record: np.array) -> np.float:
        t0 = time.perf_counter()
        prediction = model.predict(record)
        self.model_score_time = time.perf_counter() - t0
        return prediction[0]

    def _predict_onnx_model(self, model, record: np.array) -> np.float:
        t0 = time.perf_counter()
        input_name = model.get_inputs()[0].name
        label_name = model.get_outputs()[0].name
        prediction = model.run([label_name], {input_name: record})[0]
        self.model_score_time = time.perf_counter() - t0
        return prediction[0, 0]