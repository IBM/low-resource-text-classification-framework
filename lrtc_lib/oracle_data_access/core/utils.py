# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import ujson as json
import os
import random
from typing import Mapping

from lrtc_lib.definitions import ROOT_DIR
from lrtc_lib.definitions import PROJECT_PROPERTIES
from lrtc_lib.data_access.core.data_structs import nested_default_dict, Label

gold_labels_per_dataset: (str, nested_default_dict()) = None  # (dataset, URIs -> categories -> Label)


def get_gold_labels_dump_dir():
    return os.path.join(ROOT_DIR, 'data', 'oracle_access_dumps')


def get_labels_dump_filename(dataset_name: str):
    return os.path.join(get_gold_labels_dump_dir(), dataset_name + '.json')


def get_gold_labels(dataset_name: str,  category_name: str = None) -> Mapping[str, Mapping[str, Label]]:
    """
    :param dataset_name: the name of the dataset from which the gold labels should be retrieved
    :param category_name: the name of the category for which label information is needed. Default is None, meaning all
    categories.
    :return: # URIs -> categories -> Label
    """
    global gold_labels_per_dataset

    if gold_labels_per_dataset is None or gold_labels_per_dataset[0] != dataset_name:  # not in memory
        if os.path.exists(get_labels_dump_filename(dataset_name)):  # try to load from disk
            with open(get_labels_dump_filename(dataset_name)) as json_file:
                text_and_gold_labels_encoded = json_file.read()
            simplified_dict = json.loads(text_and_gold_labels_encoded)
            labels_dict = {k: {category: Label(**label_dict) for category, label_dict in v.items()}
                           for k, v in simplified_dict.items()}
            gold_labels_per_dataset = (dataset_name, labels_dict)
        else:  # or create an empty in-memory
            gold_labels_per_dataset = (dataset_name, nested_default_dict())

    uri_categories_and_labels_map = gold_labels_per_dataset[1]
    if category_name is not None:
        data_view_func = PROJECT_PROPERTIES["data_view_func"]
        uri_categories_and_labels_map = data_view_func(category_name, uri_categories_and_labels_map)
    return uri_categories_and_labels_map


def sample(dataset_name: str, category_name: str, sample_size: int, random_seed: int, restrict_label: str = None):
    """
    return a sample of TextElements uris, for the given category in the given dataset, with their gold labels
    information. If restrict_label is provided - only TextElements with that label will be included.

    :param dataset_name: the name of the dataset from which TextElements should be retrieved
    :param category_name: the name of the category whose label information is the target of this sample
    :param sample_size: how many TextElements should be sampled
    :param random_seed: a seed for the Random being used for sampling
    :param restrict_label: restrict returning TextElements to elements with the given label.
    Default is None - do not avoid any label, i.e. sample from all TextElements.
    :return: a list of tuples of TextElement uri and a dictionary of categories to Labels.
    """
    if sample_size <= 0:
        raise ValueError(f'There is no logic in sampling {sample_size} elements')

    gold_labels = get_gold_labels(dataset_name, category_name)
    gold_label_tuples = [(uri, label_dict) for uri, label_dict in gold_labels.items()]

    # restrict by label
    if restrict_label is not None:
        gold_label_tuples = [t for t in gold_label_tuples if restrict_label in t[1][category_name].labels]

    # sample
    random.Random(random_seed).shuffle(gold_label_tuples)
    return gold_label_tuples[:min(sample_size, len(gold_label_tuples))]
