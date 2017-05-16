import pandas as pd
import numpy as np
import sys, os


class DatasetHelper(object):
    """Summary of class here.

    Longer class information....

    Attributes:
        name: A boolean indicating if we like SPAM or not.
        dataset: An integer count of the eggs we have laid.
    """
    def __init__(train_file, test_file, idx_col, target):
        source_data = load_data(train_file, test_file, idx_col)
        self.data = create_dataset(source_data, target)


    def load_data(train_path, test_path, idx_col):
        # Load train/test datasets as Pandas DataFrames
        df_train = pd.read_csv(train_path, index_col=idx_col)
        df_test = pd.read_csv(test_path, index_col=idx_col)

        # Create a combined DataFrame by merging train/test sets
        df_combined = pd.concat([df_train, df_test], axis=0)

        # Rename column names: .columns -> lowercase -> snake_case
        df_combined.columns = (df_combined.columns
                               .str.replace('(.)([A-Z][a-z]+)', r'\1_\2')
                               .str.lower())

        # Initialize "Data Store" using nd-DataFrame
        store = pd.DataFrame()

        # Add Source Data to Store
        store.source = pd.DataFrame()
        store.source.train = df_train
        store.source.test = df_test
        store.dataset = df_combined

        store.train_idx = df_train.index
        store.test_idx = df_test.index
        store.combined_idx = df_combined.index
        return store


    def df_to_xy(df, y):
        data = pd.DataFrame()
        data.features = df.drop(y, axis=1)
        data.target = df[y]
        return data


    def create_dataset(data, target):
        df = data.dataset
        data.dataset_idx = df.index
        data.target = target
        data.train = df_to_xy(df.loc[data.test_idx], y=target)
        data.test = df_to_xy(df.loc[data.train_idx], y=target)
        data.combined = df_to_xy(df.loc[data.dataset_idx], y=target)
        return data


    def save_dataset(name, data, path):
        # Initialize "Data Store" using HDFStore
        if to_HDFStore:
            store = pd.HDFStore(path)
            store[name] = data
            return store.titanic
        else:
            return store