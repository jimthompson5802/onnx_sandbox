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
        self.performance_fp = performance_fp
        self.performance_file_wrote_first_record = False
        self.test_scenario = test_scenario

        # performance metrics
        self.runtime_metrics = {}
        self.benchmark_process = psutil.Process(os.getpid())
        # self.model_load_time = 0.0
        # self.model_score_time = 0.0
        # self.predicted_score = np.nan
        # self.county_id = ''
        # self.model_memory_size_mb = 0
        # self.record_id = -1

    def clear_stats(self):
        self.runtime_metrics = {}

    def write_performance_data(self):
        perf_df = pd.DataFrame([self.runtime_metrics])
        if self.performance_file_wrote_first_record:
            perf_df.to_csv(self.performance_fp, header=False, index=False, mode='a')
        else:
            perf_df.to_csv(self.performance_fp, header=True, index=False)
            self.performance_file_wrote_first_record = True

    def score_one_record(self, county_id: str, record_id: int, record: np.array):
        self.runtime_metrics['county_id'] = county_id
        self.runtime_metrics['record_id'] = record_id
        self.runtime_metrics['test_scenario'] = self.test_scenario
        if self.model_type == SKLEARN:
            model: RandomForestRegressor = self._retrieve_sklearn_model(county_id)
            self.runtime_metrics['predicted_score'] = self._predict_sklearn_model(model, record)
        else:
            model: InferenceSession = self._retrieve_onnx_model(county_id)
            self.runtime_metrics['predicted_score'] = self._predict_onnx_model(model, record)

        # record performance
        self.write_performance_data()
        self.clear_stats()


    def _retrieve_sklearn_model(self, county_id: str) -> RandomForestRegressor:
        # retrieve model from persistent storage
        t0 = time.perf_counter()
        with open(os.path.join(self.models_dir, county_id + '.pkl'), 'rb') as f:
            rf_pkl_model = pickle.load(f)
        self.runtime_metrics['model_load_time_ms'] = (time.perf_counter() - t0) * 1000
        return rf_pkl_model

    def _retrieve_onnx_model(self, county_id: str) -> InferenceSession:
        # retrieve model from file
        t0 = time.perf_counter()
        sess = rt.InferenceSession(os.path.join(self.models_dir, county_id + '.onnx'))
        self.runtime_metrics['model_load_time_ms'] = (time.perf_counter() - t0) * 1000
        return sess

    def _predict_sklearn_model(self, model, record: np.array) -> np.float:
        t0 = time.perf_counter()
        prediction = model.predict(record)
        self.runtime_metrics['model_score_time_ms'] = (time.perf_counter() - t0) * 1000
        self.runtime_metrics['model_process_rss_mb'] = self.benchmark_process.memory_info().rss / (1024 * 1024)
        return prediction[0]

    def _predict_onnx_model(self, model, record: np.array) -> np.float:
        t0 = time.perf_counter()
        input_name = model.get_inputs()[0].name
        label_name = model.get_outputs()[0].name
        prediction = model.run([label_name], {input_name: record})[0]
        self.runtime_metrics['model_score_time_ms'] = (time.perf_counter() - t0) * 1000
        self.runtime_metrics['model_process_rss_mb'] = self.benchmark_process.memory_info().rss / (1024 * 1024)
        return prediction[0, 0]