# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import logging
from collections import Counter
from typing import Tuple

from lrtc_lib.data_access.core.data_structs import Label
from lrtc_lib.data_access import data_access_factory
from lrtc_lib.training_set_selector.train_and_dev_set_selector_api import TrainAndDevSetsSelector

data_access = data_access_factory.get_data_access()
MAX_VALUE = 10000000000


class TrainAndDevSetsSelectorAllLabeled(TrainAndDevSetsSelector):
    def get_train_and_dev_sets(self, workspace_id, train_dataset_name, category_name,
                               dev_dataset_name=None) -> Tuple[Tuple, Tuple]:

        train_data, train_counts = self.get_data_and_counts_for_labeled(workspace_id, train_dataset_name, category_name,
                                                                        remove_duplicates=True)
        dev_data, dev_counts = self.get_data_and_counts_for_labeled(workspace_id, dev_dataset_name, category_name,
                                                                    remove_duplicates=True)

        logging.info(f"using {len(train_data)} for train using dataset {train_dataset_name}" +
                     (f" and {len(dev_data)} for dev using dataset {dev_dataset_name}" if dev_data is not None
                      else " with no dev dataset"))

        return (train_data, train_counts), (dev_data, dev_counts)

    def get_data_and_counts_for_labeled(self, workspace_id, dataset_name, category_name, remove_duplicates=False):
        if dataset_name is None:
            return None, None
        labeled_sample = data_access.sample_labeled_text_elements(workspace_id=workspace_id, dataset_name=dataset_name,
                                                                  category_name=category_name, sample_size=MAX_VALUE,
                                                                  remove_duplicates=remove_duplicates)["results"]
        labels = [element.category_to_label[category_name].labels for element in labeled_sample]
        labels = [item for subset in labels for item in subset]  # flatten list of sets
        counts = dict(Counter(labels))

        return labeled_sample, counts


class TrainAndDevSetsSelectorAllLabeledPlusUnlabeledAsWeakNegative(TrainAndDevSetsSelectorAllLabeled):
    """
    Use unlabeled samples as negative, only meant to be used in a binary classifier
    """
    def __init__(self, negative_ratio=None):
        """
        @param negative_ratio: number of negative samples per positive (None means all negatives)
        """
        from lrtc_lib.orchestrator.orchestrator_api import LABEL_POSITIVE, LABEL_NEGATIVE
        self.negative_ratio = negative_ratio
        self.neg_label = LABEL_NEGATIVE
        self.pos_label = LABEL_POSITIVE

    def get_train_and_dev_sets(self, workspace_id, train_dataset_name, category_name,
                               dev_dataset_name=None) -> Tuple[Tuple, Tuple]:

        train_data, train_counts = self.get_data_and_counts_for_labeled(workspace_id, train_dataset_name, category_name,
                                                                        remove_duplicates=True)
        dev_data, dev_counts = self.get_data_and_counts_for_labeled(workspace_id, dev_dataset_name, category_name,
                                                                    remove_duplicates=True)

        required_number_of_unlabeled_as_neg = MAX_VALUE
        if self.negative_ratio is not None:
            # reduce the number of samples that were labeled as negatives from the requested number of negatives
            required_number_of_unlabeled_as_neg = \
                max(0, self.negative_ratio * train_counts[self.pos_label] - train_counts.get(self.neg_label, 0))
            if required_number_of_unlabeled_as_neg > 0:
                logging.info(f"Trying to add {required_number_of_unlabeled_as_neg} to meet ratio of "
                             f"{self.negative_ratio} negatives per positive")
        else:
            logging.info(f"using all unlabeled elements as negatives")

        if required_number_of_unlabeled_as_neg > 0:
            unlabeled_sample = \
                data_access.sample_unlabeled_text_elements(workspace_id=workspace_id, dataset_name=train_dataset_name,
                                                           category_name=category_name,
                                                           sample_size=required_number_of_unlabeled_as_neg,
                                                           remove_duplicates=True)
            for element in unlabeled_sample['results']:
                element.category_to_label = {category_name: Label(self.neg_label, {})}
                train_data.append(element)
            train_counts["weak_"+self.neg_label] = len(unlabeled_sample['results'])

        logging.info(f"using {len(train_data)} for train using dataset {train_dataset_name}" +
                     (f" and {len(dev_data)} for dev using dataset {dev_dataset_name}" if dev_data is not None
                      else " with no dev dataset"))

        return (train_data, train_counts), (dev_data, dev_counts)
