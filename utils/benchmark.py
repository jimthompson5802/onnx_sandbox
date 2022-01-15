import time
from abc import abstractmethod

import joblib
import os

import pickle
import psutil
import pandas as pd
import numpy as np

class BenchmarkDriver:
    def __init__(
            self,
            model_object_type: str,
            models_dir: str,
            performance_fp: str,
            test_scenario: str
    ) -> None:
        self.model_object_type = model_object_type
        self.model_object_extension = '.' + model_object_type
        self.models_dir = models_dir
        self.performance_fp = performance_fp
        self.performance_file_wrote_first_record = False
        self.test_scenario = test_scenario

        # performance metrics
        self.runtime_metrics = {}
        self.benchmark_process = psutil.Process(os.getpid())

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
        model_object_fp = os.path.join(self.models_dir, county_id + self.model_object_extension)
        model = self.retrieve_model(model_object_fp)
        self.score_model(model, record)

        # record performance
        self.write_performance_data()
        self.clear_stats()


    def retrieve_model(self, model_fp: str):
        # retrieve model from persistent storage
        t0 = time.perf_counter()
        model = self._retrieve_this_model(model_fp)
        self.runtime_metrics['model_load_time_ms'] = (time.perf_counter() - t0) * 1000
        return model


    def score_model(self, model, record: np.array):
        t0 = time.perf_counter()
        prediction = self._score_this_model(model, record)
        self.runtime_metrics['model_score_time_ms'] = (time.perf_counter() - t0) * 1000
        self.runtime_metrics['model_process_rss_mb'] = self.benchmark_process.memory_info().rss / (1024 * 1024)
        return prediction
    
    @abstractmethod
    def _retrieve_this_model(self, model_object_fp):
        raise NotImplementedError("Missing implemnetation for '_retrieve_this_model()'")
        
    @abstractmethod
    def _score_this_model(self, model, record):
        raise NotImplementedError("Missing implementation for '_score_this_model()'")


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