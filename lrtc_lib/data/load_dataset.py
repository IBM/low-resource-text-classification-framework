# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import logging

from lrtc_lib.data_access import single_dataset_loader
from lrtc_lib.data_access.processors.dataset_part import DatasetPart
from lrtc_lib.oracle_data_access import gold_labels_loader

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')


def load(dataset: str, force_new: bool = False):
    for part in DatasetPart:
        dataset_name = dataset + '_' + part.name.lower()
        # load dataset (generate Documents and TextElements)
        if force_new:
            single_dataset_loader.clear_all_saved_files(dataset_name)
        single_dataset_loader.load_dataset(dataset_name, force_new)
        # load gold labels
        if force_new:
            gold_labels_loader.clear_gold_labels_file(dataset_name)
        gold_labels_loader.load_gold_labels(dataset_name, force_new)
        logging.info('-' * 60)


if __name__ == '__main__':
    dataset_name = 'polarity'
    load(dataset=dataset_name)