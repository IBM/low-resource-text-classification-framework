# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import dataclasses
import os
import logging
import ujson as json
import ast
import pandas as pd

from collections import defaultdict
from typing import Sequence, Iterable, Tuple
from enum import Enum
from pathlib import Path

import lrtc_lib.data_access.core.utils as utils
from lrtc_lib.data_access.core.data_structs import Document, TextElement, Label

'''
Under the DataAccessInMemory implementation of DataAccessApi, information about labels and TextElements are stored both
in the file system and in memory, in the variables "ds_in_memory" and "labels_in_memory" below.
Note that while label information changes over time (as labels are added, altered etc.), the TextElement information 
inside "ds_in_memory" will generally not change after a dataset is loaded.

===ds_in_memory===
maps dataset_name -> pandas DataFrame containing all the dataset sentences

===labels_in_memory===
maps workspace_id -> dataset name -> URIs -> categories -> labels and info
'''
ds_in_memory = defaultdict(pd.DataFrame)
labels_in_memory = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
clusters_in_memory = defaultdict(tuple)

random_seeds = {}


class LabeledStatus(Enum):
    UNLABELED = 0
    LABELED = 1
    ALL = 2


def get_ds_in_memory(dataset_name, remove_duplicates=False):
    global ds_in_memory
    if dataset_name not in ds_in_memory:
        dataset_file_path = utils.get_dataset_dump_filename(dataset_name)
        if os.path.isfile(dataset_file_path):
            df = pd.read_csv(dataset_file_path)
            # convert value of TextElement fields to their proper formats
            df = df.where(pd.notnull(df), None)
            for field in ['span', 'category_to_label', 'metadata']:
                df[field] = [ast.literal_eval(x) if x is not None else {} for x in df[field].values]
        else:
            raise Exception(f'Cannot find the dataset "{dataset_name}" dump file {dataset_file_path}.')
        ds_in_memory[dataset_name] = df
        load_duplicate_clusters(dataset_name)  # construct clusters
    res = ds_in_memory[dataset_name]
    if remove_duplicates:
        return res[res['rep_uri'].apply(lambda x: x == "self")]
    return res


def add_cluster_info(df):
    clean_to_rep = dict()
    df['rep_uri'] = None
    for index, row in df.iterrows():
        clean_text = row["text"]
        if clean_text in clean_to_rep:
            row["rep_uri"] = clean_to_rep[clean_text]
        else:
            clean_to_rep[clean_text] = row["uri"]
            row["rep_uri"] = "self"
    return df


def load_duplicate_clusters(dataset_name):
    if dataset_name not in clusters_in_memory:
        clusters = {}
        all_text_elements_df = get_ds_in_memory(dataset_name).copy()
        uri_to_rep = {x: y for x, y in zip(all_text_elements_df["uri"], all_text_elements_df["rep_uri"])}
        if len(all_text_elements_df["rep_uri"].unique()) > 1:
            clusters = {key: list(group["uri"]) for key, group in all_text_elements_df.groupby(by="rep_uri")}
            logging.info(f"Loaded {len(clusters)} duplicate clusters")
        clusters_in_memory[dataset_name] = (clusters, uri_to_rep)
    return clusters_in_memory[dataset_name]


def get_labels(workspace_id, dataset_name):
    global labels_in_memory
    if workspace_id not in labels_in_memory or dataset_name not in labels_in_memory[workspace_id]:
        file_path = utils.get_workspace_labels_dump_filename(workspace_id, dataset_name)
        if os.path.isfile(file_path):
            # Read dict from disk
            with open(file_path) as f:
                labels_encoded = f.read()
            simplified_dict = json.loads(labels_encoded)
            labels_dict = defaultdict(lambda: defaultdict(dict))
            labels_dict.update({k: {category: Label(**label_dict) for category, label_dict in v.items()}
                                for k, v in simplified_dict.items()})
            labels_in_memory[workspace_id][dataset_name] = labels_dict
        else:
            # Save empty dict to disk
            os.makedirs(Path(file_path).parent, exist_ok=True)
            empty_dict_encoded = json.dumps(labels_in_memory[workspace_id][dataset_name])
            with open(file_path, 'w') as f:
                f.write(empty_dict_encoded)
    return labels_in_memory[workspace_id][dataset_name]


def add_sentences_to_dataset_in_memory(dataset_name, sentences: Iterable[TextElement]):
    global ds_in_memory
    dicts = [dataclasses.asdict(s) for s in sentences]
    new_sentences_df = pd.DataFrame(dicts)
    new_sentences_df = add_cluster_info(new_sentences_df)
    if dataset_name in ds_in_memory:
        ds_in_memory[dataset_name] = ds_in_memory[dataset_name].append(new_sentences_df, sort=False)
    else:
        ds_in_memory[dataset_name] = new_sentences_df
    ds_in_memory[dataset_name].to_csv(utils.get_dataset_dump_filename(dataset_name), index=False)


def add_labels_info_for_doc(workspace_id, dataset_name, doc: Document):
    labels_info_for_workspace = get_labels(workspace_id, dataset_name)
    for elem in doc.text_elements:
        if elem.uri in labels_info_for_workspace:
            elem.category_to_label.update(labels_info_for_workspace[elem.uri])
    return doc


def add_labels_info_for_text_elements(workspace_id, dataset_name, text_elements: Sequence[TextElement]):
    labels_info_for_workspace = get_labels(workspace_id, dataset_name)
    for elem in text_elements:
        if elem.uri in labels_info_for_workspace:
            elem.category_to_label.update(labels_info_for_workspace[elem.uri])
    return text_elements


def filter_by_labeled_status(df: pd.DataFrame, category_name: str, labeled_status: LabeledStatus):
    """
    :param df:
    :param category_name:
    :param labeled_status: unlabeled, labeled or all
    :return:
    """
    if labeled_status == LabeledStatus.UNLABELED:
        df = df[df.apply(lambda row: category_name not in row.category_to_label, axis=1)]

    elif labeled_status == LabeledStatus.LABELED:
        df = df[df.apply(lambda row: category_name in row.category_to_label, axis=1)]

    return df


def filter_by_query(df: pd.DataFrame, query):
    if query:
        df = df[df.text.str.contains(query, na=False)]
    return df


def filter_by_query_and_label_status(df: pd.DataFrame, category_name: str, labeled_status: LabeledStatus, query: str):
    df = filter_by_labeled_status(df, category_name, labeled_status)
    df = filter_by_query(df, query)
    return df


def get_text_elements(dataset_name: str, uris: Iterable) -> Sequence[TextElement]:
    corpus_df = get_ds_in_memory(dataset_name)
    uris = list(uris)
    corpus_df = corpus_df.loc[corpus_df['uri'].isin(uris)]
    text_elements = [TextElement(*t) for t in
                     corpus_df[TextElement.get_field_names()].itertuples(index=False, name=None)]
    return text_elements


def sample_text_elements(workspace_id: str, dataset_name: str, sample_size: int, filter_func, remove_duplicates=False,
                         random_state=None) -> Tuple[Sequence, int]:
    """

    :param sample_size: if None, return all elements without sampling
    :param workspace_id: if None no labels info would be used or output
    :param dataset_name:
    :param filter_func:
    :param remove_duplicates:
    :param random_state:
    """

    corpus_df = get_ds_in_memory(dataset_name, remove_duplicates).copy()
    random_state = random_state if random_state else 0
    if workspace_id:
        random_state = sum([ord(c) for c in workspace_id]) + random_state
        labels_dict = get_labels(workspace_id, dataset_name).copy()
        corpus_df['category_to_label'] = [dict(labels_dict[u]) for u in corpus_df['uri']]
    corpus_df = filter_func(corpus_df)
    hit_count = len(corpus_df)
    if sample_size and hit_count > sample_size:
        corpus_df = corpus_df.sample(n=sample_size, random_state=random_state)
    result_text_elements = [TextElement(*t) for t in
                            corpus_df[TextElement.get_field_names()].itertuples(index=False, name=None)]
    return result_text_elements, hit_count


def clear_labels_in_memory():
    global labels_in_memory
    labels_in_memory = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))


def save_labels_data(dataset_name, workspace_id):
    file_path = utils.get_workspace_labels_dump_filename(workspace_id, dataset_name)
    os.makedirs(Path(file_path).parent, exist_ok=True)
    labels = labels_in_memory[workspace_id][dataset_name]
    simplified_labels = {k: {str(category): label.to_dict() for category, label in v.items()}
                         for k, v in labels.items()}
    labels_in_memory_encoded = json.dumps(simplified_labels)
    with open(file_path, 'w') as f:
        f.write(labels_in_memory_encoded)
