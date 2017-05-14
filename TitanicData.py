# coding=utf-8

# Import necessary packages (Pandas, NumPy, etc.)
import pandas as pd
import numpy as np

# Set file paths for Titanic data (Source: Kaggle)
filepath_train = 'Data/train.csv'
filepath_test = 'Data/test.csv'

# Load train/test datasets as Pandas DataFrames
df_train = pd.read_csv('Data/train.csv', index_col='PassengerId')
df_test = pd.read_csv('Data/test.csv', index_col='PassengerId')
