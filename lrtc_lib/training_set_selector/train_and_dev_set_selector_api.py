# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import abc
from enum import Enum
from typing import Tuple


class TrainingSetSelectionStrategy(Enum):
    ALL_LABELED = 0
    ALL_LABELED_PLUS_ALL_UNLABELED_AS_NEGATIVE = 1
    ALL_LABELED_PLUS_UNLABELED_AS_NEGATIVE_EQUAL_RATIO = 2
    ALL_LABELED_PLUS_UNLABELED_AS_NEGATIVE_X2_RATIO = 3
    ALL_LABELED_PLUS_UNLABELED_AS_NEGATIVE_X10_RATIO = 4


class TrainAndDevSetsSelector(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_train_and_dev_sets(self, workspace_id: str, train_dataset_name: str, category_name: str,
                               dev_dataset_name=None) -> Tuple[Tuple, Tuple]:
        """
        For a given workspace, dataset and category, prepare train and dev sets and return them.
        Returns a tuple with the format: (train_data, train_counts), (dev_data, dev_counts)
        where "data" is a list of TextElement objects (containing labels for the category), and "counts" is a
        dictionary detailing the number of elements per each label.

        :param workspace_id:
        :param train_dataset_name:
        :param category_name:
        :param dev_dataset_name:
        """
        raise NotImplementedError('get_train_and_dev_sets is not implemented in abstract class')
