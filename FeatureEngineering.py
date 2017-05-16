# coding=utf-8

# Load train/test datasets as Pandas DataFrames
df_train = pd.read_csv('Data/train.csv', index_col='PassengerId')
df_test = pd.read_csv('Data/test.csv', index_col='PassengerId')
df = pd.concat([df_train, df_test], axis=0)

# Add New Feature (family_size)
# `family_size` is calculated using the following features: `sibsp` & `parch`
df.family_size = df.sibsp + df.parch + 1

# Add New Feature (no_family)
# Feature reflects passengers that have no family members on-board:
# `family_size` equals 1 -> T/F -> 1/0 ==> `no_family`
df.no_family = np.array(df.family_size == 1).astype(int)

# Add New Feature (name_length)
# `name_length` equals string length of `name` feature.
df.name_length = df.name.str.len()

# Add New Feature (name_length_group)
# `name_length -> 4x bins ==> `name_length_group`
df.name_length_group = pd.cut(df.name_length, 4, labels=False)

# Add New Feature (deck)
# `deck` is calculated by extracting the first character of 'cabin' feature.
# If 'cabin' is null, than `deck` value for that observation is set to '?'.
df.deck = df.cabin.fillna('?').str.slice(0, 1)

'''
The resulting feature `title` is obtained by performing the 
following operations on `name`:
 (1) splitting ^*(string) on ',' & extracting 1st item
 (2) splitting ^(1) on '.' & extracting 1st item
 (3) stripping ^(2) of any leading/trailing whitespace characters
 (1) mapping ^(3) to title groups using `title_map`(dict)
'''
def kv_to_vk(d):
    return {t: k for k, v in d.items() for t in v}

title_map = kv_to_vk({
    'Officer': ['Capt', 'Col', 'Major', 'Rev'],
    'Royalty': ['Sir', 'Lady', 'the Countess',
                'Dona', 'Jonkheer', 'Don'],
    'Dr': ['Dr'], 'Master': ['Master'], 'Mr': ['Mr'],
    'Miss': ['Ms', 'Miss', 'Mlle'], 'Mrs': ['Mrs', 'Mme']})
df.title = (df.name
            .str.split(',').str.get(1)
            .str.split('.').str.get(0)
            .str.strip()
            .map(title_map))
