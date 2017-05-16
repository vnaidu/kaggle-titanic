# coding=utf-8
import DatasetHelper

()


class TitanicData(object):
    """Summary of class here.

    Longer class information....

    Attributes:
        name: A boolean indicating if we like SPAM or not.
        dataset: An integer count of the eggs we have laid.
    """

    def __init__(train_file, test_file, idx_col, target):
        train_file = 'Data/train.csv'
        test_file = 'Data/test.csv'
        idx_col = 'PassengerId'
        target = 'survived'

        data_manager = DatasetHelper(train_file, test_file, idx_col, target)
        source_data = load_data(train_file, test_file, idx_col)
        self.data = create_dataset(source_data, target)

    def GetData(self):


