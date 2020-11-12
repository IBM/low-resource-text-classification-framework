# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

from lrtc_lib.data_access.processors.process_ag_new_data import AgNewsProcessor
from lrtc_lib.data_access.processors.process_cola_data import ColaProcessor
from lrtc_lib.data_access.processors.process_isear_data import IsearProcessor
from lrtc_lib.data_access.processors.process_polarity_data import PolarityProcessor
from lrtc_lib.data_access.processors.process_trec_data import TrecProcessor
from lrtc_lib.data_access.processors.process_wiki_attack_data import WikiAttackProcessor
from lrtc_lib.data_access.processors.data_processor_api import DataProcessorAPI
from lrtc_lib.data_access.processors.dataset_part import DatasetPart
from lrtc_lib.data_access.processors.process_subjectivity_data import SubjectivityProcessor


def get_data_processor(dataset_name: str) -> DataProcessorAPI:
    """
    Return the DataProcessorAPI suitable for the given dataset.
    :return:
    """
    dataset_source, dataset_part = parse_dataset_name(dataset_name=dataset_name)
    if dataset_source == 'trec_50':
        return TrecProcessor(dataset_part=dataset_part, use_fine_grained_labels=True)
    if dataset_source == 'trec':
        return TrecProcessor(dataset_part=dataset_part, use_fine_grained_labels=False)
    if dataset_source == 'isear':
        return IsearProcessor(dataset_part=dataset_part)
    if dataset_source == 'subjectivity':
        return SubjectivityProcessor(dataset_part=dataset_part)
    if dataset_source == 'subjectivity_imbalanced_subjective':
        return SubjectivityProcessor(dataset_part=dataset_part, imbalanced_postfix='_imbalanced_subjective')
    if dataset_source == 'polarity':
        return PolarityProcessor(dataset_part=dataset_part)
    if dataset_source == 'polarity_imbalanced_positive':
        return PolarityProcessor(dataset_part=dataset_part, imbalanced_postfix='_imbalanced_positive')
    if dataset_source == 'ag_news':
        return AgNewsProcessor(dataset_part=dataset_part)
    if dataset_source == 'ag_news_imbalanced_1':
        return AgNewsProcessor(dataset_part=dataset_part, imbalanced_postfix='_imbalanced_1')
    if dataset_source == 'ag_news_imbalanced_2':
        return AgNewsProcessor(dataset_part=dataset_part, imbalanced_postfix='_imbalanced_2')
    if dataset_source == 'cola':
        return ColaProcessor(dataset_part=dataset_part)
    if dataset_source == 'wiki_attack':
        return WikiAttackProcessor(dataset_part=dataset_part)
    else:
        raise ValueError(f'I cannot find a data processor for dataset {dataset_source}')


def parse_dataset_name(dataset_name: str) -> (str, str):
    """
    Split the string of the dataset name into two parts: dataset source name (e.g., cnc_in_domain)
    and dataset part (e.g., train).
    :param dataset_name:
    :return: dataset source name (e.g., cnc_in_domain) and dataset part (e.g., train).
    """
    name_parts = dataset_name.rsplit('_', 1)
    dataset_source = name_parts[0]
    dataset_part = DatasetPart[name_parts[1].upper()]
    return dataset_source, dataset_part
