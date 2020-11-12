# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import random
import unittest
from typing import List

import lrtc_lib.data_access.data_access_factory as data_access_factory
import lrtc_lib.data_access.single_dataset_loader as ds_loader
from lrtc_lib.data_access.core.data_structs import Document, TextElement, Label
from lrtc_lib.data_access.core.utils import URI_SEP
from lrtc_lib.data_access.data_access_in_memory import DataAccessInMemory
from lrtc_lib.orchestrator.orchestrator_api import LABEL_POSITIVE, LABEL_NEGATIVE

data_access: DataAccessInMemory = data_access_factory.get_data_access()


def generate_simple_doc(dataset_name, doc_id=0, add_duplicate=False):
    sentences = ['Document Title is Super Interesting', 'First sentence is not that attractive.',
                 'The second one is a bit better.', 'Last sentence offers a promising view for the future!']
    if add_duplicate:
        sentences.append('First sentence  is not that attractive,')
    text_elements = []
    start_span = 0
    for idx, sentence in enumerate(sentences):
        end_span = start_span + len(sentence)
        text_elements.append(TextElement(uri=URI_SEP.join([dataset_name, str(doc_id), str(idx)]), text=sentence,
                                         span=[(start_span, end_span)], metadata={}, category_to_label={}))
        start_span = end_span + 1

    doc = Document(uri=dataset_name + URI_SEP + str(doc_id), text_elements=text_elements, metadata={})
    return doc


def generate_corpus(dataset_name, num_of_documents=1, add_duplicate=False):
    ds_loader.clear_all_saved_files(dataset_name)
    docs = [generate_simple_doc(dataset_name, doc_id, add_duplicate) for doc_id in range(0, num_of_documents)]
    data_access.add_documents(dataset_name=dataset_name, documents=docs)
    return docs


def generate_random_texts_and_labels(doc: Document, num_sentences_to_label: int, categories: List[str]):
    sentences_and_labels = []
    text_elements_to_label = random.sample(doc.text_elements, min(num_sentences_to_label, len(doc.text_elements)))
    for elem in text_elements_to_label:
        categories_to_label = random.sample(categories, random.randint(0, len(categories)))
        labels = {cat: Label(labels=LABEL_POSITIVE, metadata={}) if cat in categories_to_label else Label(
            labels=LABEL_NEGATIVE, metadata={}) for cat in categories}
        sentences_and_labels.append((elem.uri, labels))
    return sentences_and_labels


def add_labels_to_doc(doc: Document, category: str):
    sentences_and_labels = []
    for elem in doc.text_elements:
        labels = {category: Label(labels=LABEL_POSITIVE, metadata={})}
        sentences_and_labels.append((elem.uri, labels))
    return sentences_and_labels


class TestDataAccessInMemory(unittest.TestCase):

    def test_add_documents_and_get_documents(self):
        dataset_name = self.test_add_documents_and_get_documents.__name__ + '_dump'
        doc = generate_corpus(dataset_name)[0]
        doc_in_memory = data_access.get_documents(dataset_name, [doc.uri])[0]
        # compare all fields
        diffs = [(field, getattr(doc_in_memory, field), getattr(doc, field)) for field in
                 Document.__annotations__ if not getattr(doc_in_memory, field) == getattr(doc, field)]
        self.assertEqual(0, len(diffs))
        ds_loader.clear_all_saved_files(dataset_name)

    def test_set_labels_and_get_documents_with_labels_info(self):
        workspace_id = 'test_set_labels'
        dataset_name = self.test_set_labels_and_get_documents_with_labels_info.__name__ + '_dump'
        categories = ['cat_' + str(i) for i in range(3)]
        doc = generate_corpus(dataset_name)[0]
        texts_and_labels_list = generate_random_texts_and_labels(doc, 5,
                                                                 categories)  # [(uri, {category: Label})]
        data_access.set_labels(workspace_id, texts_and_labels_list)

        doc_with_labels_info = data_access.get_documents_with_labels_info(workspace_id, dataset_name, [doc.uri])
        texts_and_labels_dict = dict(texts_and_labels_list)
        for text in doc_with_labels_info[0].text_elements:
            if text.uri in texts_and_labels_dict:
                self.assertDictEqual(text.category_to_label, texts_and_labels_dict[text.uri])
            else:
                self.assertDictEqual(text.category_to_label, {})
        ds_loader.clear_all_saved_files(dataset_name)

    def test_unset_labels(self):
        workspace_id = 'test_unset_labels'
        dataset_name = self.test_set_labels_and_get_documents_with_labels_info.__name__ + '_dump'
        category = "cat1"
        doc = generate_corpus(dataset_name)[0]
        texts_and_labels_list = add_labels_to_doc(doc, category)
        data_access.set_labels(workspace_id, texts_and_labels_list)

        labels_count = data_access.get_label_counts(workspace_id, dataset_name, category)
        self.assertGreater(labels_count['true'], 0)
        data_access.unset_labels(workspace_id, category, [x[0] for x in texts_and_labels_list])
        labels_count_after_unset = data_access.get_label_counts(workspace_id, dataset_name, category)
        self.assertEqual(0, labels_count_after_unset["true"])
        ds_loader.clear_all_saved_files(dataset_name)

    def test_get_all_document_uris(self):
        dataset_name = self.test_get_all_document_uris.__name__ + '_dump'
        docs = generate_corpus(dataset_name, random.randint(1, 10))
        docs_uris_in_memory = data_access.get_all_document_uris(dataset_name)
        docs_uris_expected = [doc.uri for doc in docs]
        self.assertSetEqual(set(docs_uris_expected), set(docs_uris_in_memory))
        ds_loader.clear_all_saved_files(dataset_name)

    def test_get_all_text_elements_uris(self):
        dataset_name = self.test_get_all_text_elements_uris.__name__ + '_dump'
        docs = generate_corpus(dataset_name, random.randint(1, 10))
        text_elements_uris_in_memory = data_access.get_all_text_elements_uris(dataset_name)
        text_elements_uris_expected = [text.uri for doc in docs for text in doc.text_elements]
        self.assertSetEqual(set(text_elements_uris_expected), set(text_elements_uris_in_memory))
        ds_loader.clear_all_saved_files(dataset_name)

    def test_get_all_text_elements(self):
        dataset_name = self.test_get_all_text_elements.__name__ + '_dump'
        docs = generate_corpus(dataset_name, random.randint(1, 10))
        text_elements_found = data_access.get_all_text_elements(dataset_name)
        text_elements_found.sort(key=lambda t: t.uri)
        text_elements_expected = [text for doc in docs for text in doc.text_elements]
        text_elements_expected.sort(key=lambda t: t.uri)
        self.assertListEqual(text_elements_expected, text_elements_found)
        ds_loader.clear_all_saved_files(dataset_name)

    def test_sample_text_elements(self):
        dataset_name = self.test_sample_text_elements.__name__ + '_dump'
        sample_size = 5
        generate_corpus(dataset_name, 10)
        sampled_texts_res = data_access.sample_text_elements(dataset_name, sample_size)
        self.assertEqual(sample_size, len(sampled_texts_res['results']))

        sample_all = 10 ** 100  # a huge sample_size to sample all elements
        sampled_texts_res = data_access.sample_text_elements(dataset_name, sample_all)
        self.assertEqual(sampled_texts_res['hit_count'], len(sampled_texts_res['results']),
                         f'the number of sampled elements does not equal to the hit count, '
                         f'even though asked to sample all.')
        self.assertEqual(len(data_access.get_all_text_elements_uris(dataset_name)), sampled_texts_res['hit_count'],
                         f'the hit count does not equal to the total number of element uris in the dataset, '
                         f'even though asked to sample all.')
        # assert no labels were added
        self.assertDictEqual(sampled_texts_res['results'][0].category_to_label, {})
        ds_loader.clear_all_saved_files(dataset_name)

    def test_sample_text_elements_with_labels_info(self):
        workspace_id = 'test_sample_text_elements_with_labels_info'
        dataset_name = self.test_sample_text_elements_with_labels_info.__name__ + '_dump'
        sample_all = 10 ** 100  # a huge sample_size to sample all elements
        docs = generate_corpus(dataset_name, 5)
        # add labels info for a single doc
        selected_doc = docs[0]
        texts_and_labels_list = generate_random_texts_and_labels(selected_doc, 5, ['Autobots', 'Decepticons'])
        data_access.set_labels(workspace_id, texts_and_labels_list)
        texts_and_labels_dict = dict(texts_and_labels_list)

        sampled_texts_res = data_access.sample_text_elements_with_labels_info(workspace_id, dataset_name, sample_all)
        for doc_text in selected_doc.text_elements:
            sampled_text = [sampled for sampled in sampled_texts_res['results'] if sampled.uri == doc_text.uri]
            self.assertEqual(1, len(sampled_text))
            if sampled_text[0].uri in texts_and_labels_dict:
                self.assertDictEqual(sampled_text[0].category_to_label, texts_and_labels_dict[sampled_text[0].uri],
                                     f'for text {doc_text}')
            else:
                self.assertDictEqual(sampled_text[0].category_to_label, {}, f'for text {doc_text}')
        ds_loader.clear_all_saved_files(dataset_name)

    def test_sample_unlabeled_text_elements(self):
        workspace_id = 'test_sample_unlabeled_text_elements'
        dataset_name = self.test_sample_unlabeled_text_elements.__name__ + '_dump'
        category = 'Autobots'
        sample_all = 10 ** 100  # a huge sample_size to sample all elements
        docs = generate_corpus(dataset_name, 2)
        # add labels info for a single doc
        selected_doc = docs[0]
        texts_and_labels_list = generate_random_texts_and_labels(selected_doc, 5, [category])
        data_access.set_labels(workspace_id, texts_and_labels_list)

        sampled_texts_res = data_access.sample_unlabeled_text_elements(workspace_id, dataset_name, category, sample_all)
        for sampled_text in sampled_texts_res['results']:
            self.assertDictEqual(sampled_text.category_to_label, {})
        ds_loader.clear_all_saved_files(dataset_name)

    def test_sample_labeled_text_elements(self):
        workspace_id = 'test_sample_labeled_text_elements'
        dataset_name = self.test_sample_labeled_text_elements.__name__ + '_dump'
        category = 'Decepticons'
        sample_all = 10 ** 100  # a huge sample_size to sample all elements
        docs = generate_corpus(dataset_name, 2)
        # add labels info for a single doc
        selected_doc = docs[0]
        texts_and_labels_list = generate_random_texts_and_labels(selected_doc, 5, [category])
        data_access.set_labels(workspace_id, texts_and_labels_list)
        texts_and_labels_dict = dict(texts_and_labels_list)

        sampled_texts_res = data_access.sample_labeled_text_elements(workspace_id, dataset_name, category, sample_all)
        self.assertEqual(len(texts_and_labels_list), len(sampled_texts_res['results']),
                         f'all and only the {len(texts_and_labels_list)} labeled elements should have been sampled.')

        for sampled_text in sampled_texts_res['results']:
            self.assertIn(sampled_text.uri, texts_and_labels_dict.keys(),
                          f'the sampled text uri - {sampled_text.uri} - was not found in the '
                          f'texts that were labeled: {texts_and_labels_dict}')
            self.assertDictEqual(sampled_text.category_to_label, texts_and_labels_dict[sampled_text.uri])
        ds_loader.clear_all_saved_files(dataset_name)

    def test_sample_by_query_text_elements(self):
        workspace_id = 'test_sample_by_query_text_elements'
        dataset_name = self.test_sample_by_query_text_elements.__name__ + '_dump'
        category = 'Autobots'
        query = 'sentence'
        sample_all = 10 ** 100  # a huge sample_size to sample all elements
        doc = generate_corpus(dataset_name, 1)[0]
        # doc's elements = ['Document Title is Super Interesting', 'First sentence is not that attractive.',
        #          'The second one is a bit better.', 'Last sentence offers a promising view for the future!']
        # add labels info for a single doc
        texts_and_labels_list = [
            # 1st sent does not match query
            (doc.text_elements[0].uri, {category: Label(labels=LABEL_POSITIVE, metadata={})}),
            # 2nd sent does match query
            (doc.text_elements[1].uri, {category: Label(labels=LABEL_POSITIVE, metadata={})})]
        data_access.set_labels(workspace_id, texts_and_labels_list)

        # query + unlabeled elements
        sampled_texts_res = data_access.sample_unlabeled_text_elements(workspace_id, dataset_name, category, sample_all,
                                                                       query)
        for sampled_text in sampled_texts_res['results']:
            self.assertDictEqual(sampled_text.category_to_label, {})

        # query + labeled elements
        sampled_texts_res = data_access.sample_labeled_text_elements(workspace_id, dataset_name, category, sample_all,
                                                                     query)
        self.assertEqual(1, len(sampled_texts_res['results']),
                         f'all and only the {len(texts_and_labels_list)} labeled elements should have been sampled.')
        texts_and_labels_dict = dict(texts_and_labels_list)
        for sampled_text in sampled_texts_res['results']:
            self.assertIn(sampled_text.uri, texts_and_labels_dict.keys(),
                          f'the sampled text uri - {sampled_text.uri} - was not found in the '
                          f'texts that were labeled: {texts_and_labels_dict}')
            self.assertIn(query, sampled_text.text)
        ds_loader.clear_all_saved_files(dataset_name)

    def test_get_label_counts(self):
        workspace_id = 'test_get_label_counts'
        dataset_name = self.test_get_label_counts.__name__ + '_dump'
        category = 'Decepticons'
        docs = generate_corpus(dataset_name, 2)
        # add labels info for a single doc
        selected_doc = docs[0]
        texts_and_labels_list = generate_random_texts_and_labels(selected_doc, 5, ['Autobots'])
        if texts_and_labels_list:
            if category in texts_and_labels_list[0][1]:
                texts_and_labels_list[0][1][category].labels = frozenset(LABEL_NEGATIVE)
            else:
                texts_and_labels_list[0][1][category] = Label(labels=LABEL_NEGATIVE, metadata={})
        data_access.set_labels(workspace_id, texts_and_labels_list)

        category_label_counts = data_access.get_label_counts(workspace_id, dataset_name, category)
        for label_val, observed_count in category_label_counts.items():
            expected_count = len(
                [t for t in texts_and_labels_list if category in t[1] and label_val in t[1][category].labels])
            self.assertEqual(expected_count, observed_count, f'count for {label_val} does not match.')
        ds_loader.clear_all_saved_files(dataset_name)

    def test_get_text_elements_by_id(self):
        workspace_id = "test_get_text_elements_by_id"
        dataset_name = self.test_get_text_elements_by_id.__name__ + '_dump'
        categories = ['cat_' + str(i) for i in range(3)]
        docs = generate_corpus(dataset_name, 2)
        doc = docs[0]
        texts_and_labels_list = generate_random_texts_and_labels(doc, 5, categories)  # [(uri, {category: Label})]
        uri_to_labels = dict(texts_and_labels_list)
        data_access.set_labels(workspace_id, texts_and_labels_list)
        uris = [x.uri for doc in docs for x in doc.text_elements][0:2]
        all_elements = data_access.get_text_elements_with_labels_info(workspace_id, dataset_name, uris)

        self.assertEqual(len(uris), len(all_elements))
        self.assertEqual(uri_to_labels[all_elements[0].uri], all_elements[0].category_to_label)
        self.assertEqual(uri_to_labels[all_elements[1].uri], all_elements[1].category_to_label)
        ds_loader.clear_all_saved_files(dataset_name)

    def test_duplicates_removal(self):
        workspace_id = 'test_duplicates_removal'
        dataset_name = self.test_duplicates_removal.__name__ + '_dump'
        generate_corpus(dataset_name, 1, add_duplicate=True)
        all_elements = data_access.get_all_text_elements(dataset_name)
        all_elements2 = data_access.sample_text_elements(dataset_name, 10**6, remove_duplicates=False)['results']
        self.assertListEqual(all_elements, all_elements2)
        all_without_dups = data_access.sample_text_elements(dataset_name, 10**6, remove_duplicates=True)['results']
        self.assertEqual(len(all_elements), len(all_without_dups)+1)

        category = 'cat1'
        texts_and_labels_list = [(elem.uri, {category: Label(labels=LABEL_POSITIVE, metadata={})})
                                 for elem in all_without_dups]
        # set labels without propagating to duplicates
        data_access.set_labels(workspace_id, texts_and_labels_list, propagate_to_duplicates=False)
        labels_count = data_access.get_label_counts(workspace_id, dataset_name, category)
        self.assertEqual(labels_count[LABEL_POSITIVE], len(all_without_dups))
        # unset labels
        data_access.unset_labels(workspace_id, category, [elem.uri for elem in all_without_dups])
        labels_count = data_access.get_label_counts(workspace_id, dataset_name, category)
        self.assertEqual(labels_count[LABEL_POSITIVE], 0)
        # set labels with propagating to duplicates
        data_access.set_labels(workspace_id, texts_and_labels_list, propagate_to_duplicates=True)
        labels_count = data_access.get_label_counts(workspace_id, dataset_name, category)
        self.assertEqual(labels_count[LABEL_POSITIVE], len(all_elements))
        ds_loader.clear_all_saved_files(dataset_name)


if __name__ == "__main__":
    unittest.main()
