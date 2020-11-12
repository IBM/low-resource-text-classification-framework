# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import logging
import os
import lrtc_lib.oracle_data_access.core.utils as oracle_utils
import lrtc_lib.oracle_data_access.oracle_data_access_api as oracle
from lrtc_lib.data_access.processors.data_processor_api import DataProcessorAPI
from lrtc_lib.data_access.processors.dataset_part import DatasetPart
import lrtc_lib.data_access.processors.data_processor_factory as data_processor_factory


def load_gold_labels(dataset_name: str, force_new=False, processor_factory=data_processor_factory):
    """
    Load the gold labels of the given dataset.
    :param dataset_name: the name of the dataset to which the gold labels refer.
    :param force_new: default is false.
    :param processor_factory: a factory for DataProcessorAPI. Default is data_processor_factory.
    """

    gold_labels_file_path = oracle_utils.get_labels_dump_filename(dataset_name)
    if os.path.exists(gold_labels_file_path) and not force_new:
        logging.info(f'{dataset_name}:\t\tskipping loading gold labels as {gold_labels_file_path} exists. '
                     f'You can force a new loading by passing the parameter force_new=True')
    else:
        data_processor: DataProcessorAPI = processor_factory.get_data_processor(dataset_name)
        uris_to_gold_labels = data_processor.get_texts_and_gold_labels()
        oracle.add_gold_labels(dataset_name, uris_to_gold_labels)
        logging.info(f'{dataset_name}:\t\tloaded the gold labels of {len(uris_to_gold_labels)} TextElements '
                     f'to {gold_labels_file_path}')


def clear_gold_labels_file(dataset_name: str):
    """
    Delete the cached file of the gold labels for the given dataset.
    :param dataset_name: the name of the dataset to which the gold labels refer.
    """
    gold_labels_file = oracle_utils.get_labels_dump_filename(dataset_name)
    if os.path.isfile(gold_labels_file):
        os.remove(gold_labels_file)


if __name__ == '__main__':
    dataset_sources = ['polarity']

    for dataset_source in dataset_sources:
        for part in DatasetPart:
            dataset = dataset_source + '_' + part.name.lower()
            clear_gold_labels_file(dataset)
            load_gold_labels(dataset)
