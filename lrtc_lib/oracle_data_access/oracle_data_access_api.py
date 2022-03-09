# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import dataclasses
import os
import ujson as json
from typing import Sequence, List, Mapping, Tuple, Set

import lrtc_lib.oracle_data_access.core.utils as oracle_utils
from data_access.data_access_api import DataAccessApi
from lrtc_lib.data_access.core.data_structs import Label
from lrtc_lib.orchestrator.orchestrator_api import LABEL_POSITIVE, LABEL_NEGATIVE
from oracle_data_access.core.utils import create_gold_labels_online


def add_gold_labels(dataset_name: str, text_and_gold_labels: List[Tuple[str, Mapping[str, Label]]]):
    """
    Record the gold labels information for a given dataset.

    Will override previously existing information for this dataset.
    :param dataset_name:
    :param text_and_gold_labels: list of tuples of TextElement URI and a dict that represents a label.
        The dict keys are category names and values are Labels. For example: [(uri_1, {category_1: Label_cat_1}),
                                                                              (uri_2, {category_1: Label_cat_1,
                                                                                       category_2: Label_cat_2})]
    """
    oracle_utils.gold_labels_per_dataset = (dataset_name, dict(text_and_gold_labels))
    # Save gold labels to disk
    simplified_labels = {k: {str(category): label.to_dict() for category, label in v.items()}
                         for k, v in oracle_utils.gold_labels_per_dataset[1].items()}
    gold_labels_encoded = json.dumps(simplified_labels)
    os.makedirs(oracle_utils.get_gold_labels_dump_dir(), exist_ok=True)
    with open(oracle_utils.get_labels_dump_filename(dataset_name), 'w') as f:
        f.write(gold_labels_encoded)


def get_gold_labels(dataset_name: str, text_element_uris: Sequence[str], category_name: str = None,
                    data_access: DataAccessApi = None) -> \
        List[Tuple[str, Mapping[str, Label]]]:
    """
    Return the gold labels information for the given TextElements uris, keeping the same order, for the given dataset.
    If no gold labels information was added for this dataset, an empty dict is returned.

    :param dataset_name: the name of the dataset from which the gold labels should be retrieved
    :param text_element_uris:
    :param category_name: the name of the category for which label information is needed. Default is None, meaning all
    categories.
    :return: a list of tuples of TextElement uri and a dictionary of categories to Labels. The order of tuples is the
    same order as the order of the TextElement uris given as input.
    """

    gold_labels = oracle_utils.get_gold_labels(dataset_name, None)
    uris_missing = [uri for uri in text_element_uris
                    if uri not in gold_labels or not gold_labels[uri]]

    if uris_missing:
        create_gold_labels_online(dataset_name, category_name, uris_missing, data_access)

    gold_labels = oracle_utils.get_gold_labels(dataset_name, None)
    gold_labels_dataset = [(uri, gold_labels[uri])
                           for uri in text_element_uris
                           if uri in gold_labels and gold_labels[uri]]

    return gold_labels_dataset


def sample(dataset_name: str, category_name: str, sample_size: int, random_seed: int):
    """
    return a sample of TextElements uris, for the given category in the given dataset, with their gold labels
    information.
    :param dataset_name: the name of the dataset from which TextElements should be retrieved
    :param category_name: the name of the category whose label information is the target of this sample
    :param sample_size: how many TextElements should be sampled
    :param random_seed: a seed for the Random being used for sampling
    :return: a list of tuples of TextElement uri and a dictionary of categories to Labels.
    """

    return oracle_utils.sample(dataset_name=dataset_name, category_name=category_name, sample_size=sample_size,
                               random_seed=random_seed)


def sample_positives(dataset_name: str, category_name: str, sample_size: int, random_seed: int):
    """
    return a sample of TextElements uris with their gold labels information such that all these TextElements have a
    gold positive label for the given category in the given dataset.
    :param dataset_name: the name of the dataset from which TextElements should be retrieved
    :param category_name: the name of the category whose label information is the target of this sample
    :param sample_size: how many TextElements should be sampled
    :param random_seed: a seed for the Random being used for sampling
    :return: a list of tuples of TextElement uri and a dictionary of categories to Labels.
    """

    return oracle_utils.sample(dataset_name=dataset_name, category_name=category_name, sample_size=sample_size,
                               random_seed=random_seed, restrict_label=LABEL_POSITIVE)


def sample_negatives(dataset_name: str, category_name: str, sample_size: int, random_seed: int):
    """
    return a sample of TextElements uris with their gold labels information such that all these TextElements have a
    gold negative label for the given category in the given dataset.
    :param dataset_name: the name of the dataset from which TextElements should be retrieved
    :param category_name: the name of the category whose label information is the target of this sample
    :param sample_size: how many TextElements should be sampled
    :param random_seed: a seed for the Random being used for sampling
    :return: a list of tuples of TextElement uri and a dictionary of categories to Labels.
    """

    return oracle_utils.sample(dataset_name=dataset_name, category_name=category_name, sample_size=sample_size,
                               random_seed=random_seed, restrict_label=LABEL_NEGATIVE)


def get_all_labels(dataset_name: str) -> Set[str]:
    gold_labels = oracle_utils.get_gold_labels(dataset_name)
    return set(list(gold_labels.values())[0].keys())
