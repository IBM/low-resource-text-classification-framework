# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import os
from collections import defaultdict
from typing import List, Mapping, Tuple

import pandas as pd

from lrtc_lib.data_access.core.data_structs import Document, TextElement, Label, nested_default_dict
from lrtc_lib.data_access.processors.data_processor_api import DataProcessorAPI, METADATA_CONTEXT_KEY
from lrtc_lib.data_access.processors.dataset_part import DatasetPart
from lrtc_lib.data_access.core.utils import URI_SEP


def add_column_or_default_to_zip(current_zip, df, col_name_to_add, default_val):
    flatlist = list(zip(*current_zip))
    if col_name_to_add:
        if col_name_to_add not in df.columns:
            raise NameError(f'"{col_name_to_add}" is not one of the columns in the given DataFrame: {df.columns}')
        list_to_add = list(df[col_name_to_add])
    else:
        list_to_add = [default_val] * len(current_zip)
    flatlist.append(list_to_add)
    return list(zip(*flatlist))


class CsvProcessor(DataProcessorAPI):
    """
    A DataProcessor for corpus that is given in a csv format, one TextElement per line.

    """

    def __init__(self, dataset_name: str, dataset_part: DatasetPart, text_col: str = 'text',
                 label_col: str = 'label', context_col: str = None,
                 doc_id_col: str = None,
                 encoding: str = 'utf-8'):
        """

        :param dataset_name: the name of the processed dataset
        :param dataset_part: the part - train/dev/test - of the dataset
        :param text_col: the name of the column which holds the text of the TextElement. Default is 'text'.
        :param label_col: the name of the column which holds the label. Default is 'label'.
        :param context_col: the name of the column which provides context for the text, None if no context is available.
        Default is None.
        :param doc_id_col: column name by which text elements should be grouped into docs.
        If None all text elements would be put in a single dummy doc. Default is None.
        :param encoding: the encoding to use to read the csv raw file(s). Default is `utf-8`.

        """
        super().__init__(dataset_part)
        self.documents = []
        self.uri_category_labels = []
        self.dataset_name = dataset_name
        self.text_col = text_col
        self.label_col = label_col
        self.context_col = context_col
        self.doc_id_col = doc_id_col
        self.encoding = encoding
        self._process()

    def build_documents(self) -> List[Document]:
        """
        Process the raw data into a list of Documents. No labels information is provided.

        :rtype: List[Document]
        """
        return self.documents

    def get_texts_and_gold_labels(self) -> List[Tuple[str, Mapping[str, Label]]]:
        """
        Process the raw data into gold labels information.

        :rtype: a list of tuples of uri and a dict. The dict keys are category names and values are Labels.
        For example: [(uri_1, {category_1: Label_cat_1}),
                      (uri_2, {category_1: Label_cat_1,
                               category_2: Label_cat_2})]
        """
        return self.uri_category_labels

    def _get_train_file_path(self) -> str:
        return os.path.join(self.RAW_DATA_BASE_DIR, self.dataset_name, 'train.csv')

    def _get_dev_file_path(self) -> str:
        return os.path.join(self.RAW_DATA_BASE_DIR, self.dataset_name, 'dev.csv')

    def _get_test_file_path(self) -> str:
        return os.path.join(self.RAW_DATA_BASE_DIR, self.dataset_name, 'test.csv')

    def _get_all_categories(self):
        full_data_file = os.path.join(self.RAW_DATA_BASE_DIR, self.dataset_name, 'test.csv')
        df = pd.read_csv(full_data_file, encoding=self.encoding)
        return sorted(df[self.label_col].unique())

    def _process(self):
        if not os.path.isfile(self.get_raw_data_path()):
            raise Exception(f'{self.dataset_part.name.lower()} set file for dataset "{self.dataset_name}" not found')
        all_categories = self._get_all_categories()
        df = pd.read_csv(self.get_raw_data_path(), encoding=self.encoding)

        texts_categories_contexts_doc_ids = [(text, category) for text, category in
                                             list(zip(df[self.text_col], df[self.label_col]))]

        texts_categories_contexts_doc_ids = \
            add_column_or_default_to_zip(texts_categories_contexts_doc_ids, df, self.context_col, None)

        texts_categories_contexts_doc_ids = \
            add_column_or_default_to_zip(texts_categories_contexts_doc_ids, df, self.doc_id_col, 0)

        uri_to_category_labels = []
        prev_doc_id = None
        element_id = -1
        text_span_start = 0
        doc_uri_to_text_elements = defaultdict(list)
        for idx, (text, category, context, doc_id) in enumerate(texts_categories_contexts_doc_ids):
            if prev_doc_id is not None and prev_doc_id != doc_id:
                element_id = -1
                text_span_start = 0

            doc_uri = self.dataset_name + '_' + self.dataset_part.name.lower() + URI_SEP + str(doc_id)
            element_id += 1
            text_element_uri = doc_uri + URI_SEP + str(element_id)
            metadata = {METADATA_CONTEXT_KEY: context} if context else {}
            text_element = TextElement(uri=text_element_uri, text=text,
                                       span=[(text_span_start, (text_span_start+len(text)))], metadata=metadata,
                                       category_to_label={})
            doc_uri_to_text_elements[doc_uri].append(text_element)
            category_to_label_dict = \
                {cat: Label(labels=self.LABEL_POSITIVE, metadata={}) if cat == category
                else Label(labels=self.LABEL_NEGATIVE, metadata={}) for cat in all_categories}
            uri_to_category_labels.append((text_element_uri, category_to_label_dict))
            prev_doc_id = doc_id
            text_span_start += (len(text) + 1)

        self.documents = [Document(uri=doc_uri, text_elements=text_elements, metadata={})
                          for doc_uri, text_elements in doc_uri_to_text_elements.items()]
        self.uri_category_labels = uri_to_category_labels
