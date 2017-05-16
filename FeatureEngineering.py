import pandas as pd
import numpy as np


class TitanicFeatures(object):
    """Summary of class here.

    Longer class information....

    Attributes:
        name: A 
        dataset: A 
    """

    def __init__(self, df):
        self.get_all = self.all_features(df)

    @staticmethod
    def family_size(sibsp, parch):
        """
        Add New Feature (family_size):
        `family_size` is calculated using the following features: `sibsp` & `parch`
        :param sibsp: 
        :param parch: 
        :return: family_size
        """
        family_size = sibsp + parch + 1
        return family_size

    @staticmethod
    def no_family(family_size):
        """
        Add New Feature (family_size):
        Feature reflects passengers that have no family members on-board.
        :param family_size: 
        :return: 
        """
        # `no_family` <= [1|0] <- [T|F] <- (family_size == 1)
        no_family = np.array(family_size == 1).astype(int)
        return no_family

    @staticmethod
    def name_length(name):
        """
        Add New Feature (name_length):
        `name_length` equals string length of `name` feature.
        :param name: 
        :return: name_length
        """
        name_length = name.str.len()
        return name_length

    @staticmethod
    def name_length_group(name_length):
        """
        Add New Feature (name_length_group):
        `name_length -> 4x bins ==> `name_length_group`
        :param name_length: 
        :return: name_length_group
        """
        name_length_group = pd.cut(name_length, 4, labels=False)
        return name_length_group

    @staticmethod
    def deck(cabin):
        """
        Add New Feature (deck):
    
        :param cabin: 
        :return: deck
        """
        deck = cabin.fillna('?').str.slice(0, 1)
        return deck

    @staticmethod
    def title(name):
        """
        Add New Feature (title):
        The resulting feature `title` is obtained by performing the 
        following operations on `name`:
             (1) splitting ^*(string) on ',' & extracting 1st item
             (2) splitting ^(1) on '.' & extracting 1st item
             (3) stripping ^(2) of any leading/trailing whitespace characters
             (1) mapping ^(3) to title groups using `title_map`(dict)
        :param name: 
        :return: title
        """

        def kv_to_vk(d):
            return {t: k for k, v in d.items() for t in v}

        title_map = kv_to_vk({
            'Officer': ['Capt', 'Col', 'Major', 'Rev'],
            'Royalty': ['Sir', 'Lady', 'the Countess',
                        'Dona', 'Jonkheer', 'Don'],
            'Dr': ['Dr'], 'Master': ['Master'], 'Mr': ['Mr'],
            'Miss': ['Ms', 'Miss', 'Mlle'], 'Mrs': ['Mrs', 'Mme']})

        title = (name
                 .str.split(',').str.get(1)
                 .str.split('.').str.get(0)
                 .str.strip()
                 .map(title_map))
        return title

    @classmethod
    def all_features(cls, df):
        """
        Add all available new features.
        :param data: 
        :return: 
        """
        df['family_size'] = cls.family_size(sibsp=df.sib_sp, parch=df.parch)
        df['no_family'] = cls.no_family(family_size=df.family_size)
        df['name_length'] = cls.name_length(name=df.name)
        df['name_length_group'] = cls.name_length_group(name_length=df.name_length)
        df['deck'] = cls.deck(cabin=df.cabin)
        df['title'] = cls.title(name=df.name)
        return df
