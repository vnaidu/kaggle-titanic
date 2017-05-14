# coding=utf-8

#
# Load train/test datasets as Pandas DataFrames
df_train = pd.read_csv('Data/train.csv', index_col='PassengerId')
df_test = pd.read_csv('Data/test.csv', index_col='PassengerId')
df = pd.concat([df_train, df_test], axis=0)

