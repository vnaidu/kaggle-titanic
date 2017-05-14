# coding=utf-8

#
# Load train/test datasets as Pandas DataFrames
df_train = pd.read_csv('Data/train.csv', index_col='PassengerId')
df_test = pd.read_csv('Data/test.csv', index_col='PassengerId')
df = pd.concat([df_train, df_test], axis=0)

# Add New Feature (family_size)
# `family_size` is calculated using the following features: `sibsp` & `parch`
df.family_size = df.sibsp + df.parch + 1

