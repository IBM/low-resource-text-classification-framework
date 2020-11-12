# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import os
from typing import List, Mapping, Tuple, Sequence
from lrtc_lib.data_access.core.data_structs import Document, TextElement, Label
from lrtc_lib.data_access.core.utils import URI_SEP
from lrtc_lib.data_access.processors.data_processor_api import DataProcessorAPI
from lrtc_lib.data_access.processors.dataset_part import DatasetPart


class TrecProcessor(DataProcessorAPI):

    def __init__(self, dataset_part: DatasetPart, use_fine_grained_labels: bool):
        super().__init__(dataset_part)
        self.use_fine_grained_labels = use_fine_grained_labels
        self.documents = []
        self.uri_category_labels = []
        labels_version = '_50' if use_fine_grained_labels else ''
        self.doc_uri = 'trec' + labels_version + '_' + dataset_part.name.lower() + URI_SEP+'0'
        self.sep_for_idx = '\torig_line_number\t'
        self._process()

    def build_documents(self) -> List[Document]:
        """
        Process the raw data into a list of Documents. No labels information is provided.

        :param dataset_part: the part of the dataset to process (e.g., train/dev/test)
        :rtype: List[Document]
        """
        return self.documents

    def get_texts_and_gold_labels(self) -> List[Tuple[str, Mapping[str, Label]]]:
        """
        Process the raw data into gold labels information.

        :param dataset_part: the part of the dataset to process (e.g., train/dev/test)
        :rtype: a list of tuples of uri and a dict. The dict keys are category names and values are Labels.
        For example: [(uri_1, {category_1: Label_cat_1}),
                      (uri_2, {category_1: Label_cat_1,
                               category_2: Label_cat_2})]
        """
        return self.uri_category_labels

    def _get_train_file_path(self) -> str:
        return os.path.join(self.RAW_DATA_BASE_DIR, 'trec', 'train.txt')

    def _get_dev_file_path(self) -> str:
        return os.path.join(self.RAW_DATA_BASE_DIR, 'trec', 'dev.txt')

    def _get_test_file_path(self) -> str:
        return os.path.join(self.RAW_DATA_BASE_DIR, 'trec', 'test.txt')

    def _process(self):
        raw_data_file_path = self.get_raw_data_path()
        all_categories = self._get_all_categories()
        text_elements = []
        uri_to_category_labels = []
        with open(raw_data_file_path, 'r', encoding='latin-1') as f:
            labels_text_split = [line.rstrip().split(' ', 1) for line in f.readlines()]
        texts = [elem[1].split(self.sep_for_idx)[0] for elem in labels_text_split]
        categories_tuple = [(elem[0].split(':')[0], elem[0]) for elem in labels_text_split]
        texts_and_labels = list(zip(texts, categories_tuple))  # [(text, (coarse-grained, fine-grained))]
        for text_element_id, (text, categories) in enumerate(texts_and_labels):
            uri = self.doc_uri + URI_SEP + str(text_element_id)
            text_elements.append(TextElement(uri=uri, text=text, span=[(0, len(text))],
                                       metadata={}, category_to_label={}))
            category = categories[1] if self.use_fine_grained_labels else categories[0]
            category_to_label_dict = {cat: Label(labels=self.LABEL_POSITIVE, metadata={}) if cat == category
                                      else Label(labels=self.LABEL_NEGATIVE, metadata={})
                                      for cat in all_categories}
            uri_to_category_labels.append((uri, category_to_label_dict))
        self.documents = [Document(uri=self.doc_uri, text_elements=text_elements, metadata={})]
        self.uri_category_labels = uri_to_category_labels

    def _get_all_categories(self) -> Sequence[str]:
        test_file = os.path.join(self.RAW_DATA_BASE_DIR, 'trec', 'train.txt')

        with open(test_file, 'r') as f:
            labels = {line.rstrip().split(' ', 1)[0] for line in f.readlines()}
        if not self.use_fine_grained_labels:
            labels = {label.split(':')[0] for label in labels}
        return sorted(labels)


if __name__ == "__main__":
    dataset_part = DatasetPart.TRAIN
    proc = TrecProcessor(dataset_part, use_fine_grained_labels=True)
    docs = proc.build_documents()
    gold_labels = proc.get_texts_and_gold_labels()
    print(f'built {len(docs)} Documents and found labels for {len(gold_labels)} TextElements')
