# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import unittest
from math import floor
import random
from typing import List

import lrtc_lib.oracle_data_access.oracle_data_access_api as oracle
import lrtc_lib.oracle_data_access.gold_labels_loader as loader
from lrtc_lib.data_access.core.data_structs import Label
from lrtc_lib.data_access.core.utils import URI_SEP
from lrtc_lib.orchestrator.orchestrator_api import LABEL_POSITIVE, LABEL_NEGATIVE


def generate_random_uris_and_labels(dataset_name: str, num_texts_to_label: int, categories: List[str]):
    sentences_and_labels = []
    for i in range(num_texts_to_label):
        categories_to_label = random.sample(categories, random.randint(0, len(categories)))
        labels = {cat: Label(labels=LABEL_POSITIVE, metadata={}) if cat in categories_to_label else Label(
            labels=LABEL_NEGATIVE, metadata={}) for cat in categories}
        sentences_and_labels.append((dataset_name + URI_SEP + str(i), labels))
    return sentences_and_labels


class TestOracleDataAccess(unittest.TestCase):

    def assert_uri_labels_equal(self, uris_to_gold_labels_expected, uri_to_gold_labels_found):
        self.assertEqual(len(uris_to_gold_labels_expected), len(uri_to_gold_labels_found))
        for expected, found in zip(uris_to_gold_labels_expected, uri_to_gold_labels_found):
            self.assertEqual(expected[0], found[0])  # check uri, order should be the same
            self.assertEqual(expected[1], found[1])  # check labels dict

    def test_add_and_get_gold_labels_no_dump(self):
        num_elements_to_labels = 10
        dataset_name = self.test_add_and_get_gold_labels_no_dump.__name__

        uris_to_gold_labels_expected = generate_random_uris_and_labels(dataset_name, num_elements_to_labels,
                                                                       ['Autobots', 'Decepticons'])
        oracle.add_gold_labels(dataset_name, uris_to_gold_labels_expected)

        uris_to_retrieve = [uri for uri, labels_dict in uris_to_gold_labels_expected]
        uri_to_gold_labels_found = oracle.get_gold_labels(dataset_name, uris_to_retrieve)

        self.assert_uri_labels_equal(uris_to_gold_labels_expected, uri_to_gold_labels_found)
        loader.clear_gold_labels_file(dataset_name)

    def test_add_and_get_gold_labels_for_category_no_dump(self):
        num_elements_to_labels = 10
        dataset_name = self.test_add_and_get_gold_labels_for_category_no_dump.__name__

        target_category = 'Autobots'
        uris_to_gold_labels_expected = generate_random_uris_and_labels(dataset_name, num_elements_to_labels,
                                                                       [target_category])
        non_target_category = 'Decepticons'
        uris_to_gold_labels = [(uri, dict(labels, **{non_target_category: Label(labels=LABEL_POSITIVE, metadata={})}))
                               for uri, labels in uris_to_gold_labels_expected]

        oracle.add_gold_labels(dataset_name, uris_to_gold_labels)

        uris_to_retrieve = [uri for uri, labels_dict in uris_to_gold_labels_expected]
        uri_to_gold_labels_found = oracle.get_gold_labels(dataset_name, uris_to_retrieve, target_category)

        self.assert_uri_labels_equal(uris_to_gold_labels_expected, uri_to_gold_labels_found)
        loader.clear_gold_labels_file(dataset_name)

    def test_add_and_get_some_gold_labels_no_dump(self):
        num_elements_to_labels = 10
        sample_ratio = 0.4
        dataset_name = self.test_add_and_get_some_gold_labels_no_dump.__name__

        uris_to_gold_labels_expected = generate_random_uris_and_labels(dataset_name, num_elements_to_labels,
                                                                       ['Autobots', 'Decepticons'])
        oracle.add_gold_labels(dataset_name, uris_to_gold_labels_expected)
        uris_to_retrieve = [uri for uri, labels_dict in uris_to_gold_labels_expected]
        indices = random.sample(range(len(uris_to_retrieve)), floor(num_elements_to_labels * sample_ratio))
        uris_to_retrieve = [uris_to_retrieve[i] for i in sorted(indices)]
        uris_to_gold_labels_expected = [uri_label for uri_label in uris_to_gold_labels_expected if
                                        uri_label[0] in uris_to_retrieve]
        uri_to_gold_labels_found = oracle.get_gold_labels(dataset_name, uris_to_retrieve)

        self.assert_uri_labels_equal(uris_to_gold_labels_expected, uri_to_gold_labels_found)
        loader.clear_gold_labels_file(dataset_name)

    def test_add_and_get_gold_labels_from_dump(self):
        num_elements_to_labels = 10

        # generate gold labels for the first dataset
        dataset_name_first = self.test_add_and_get_gold_labels_from_dump.__name__ + '_a'
        uris_to_gold_labels_expected_first = generate_random_uris_and_labels(dataset_name_first, num_elements_to_labels,
                                                                             ['Autobots', 'Decepticons'])
        oracle.add_gold_labels(dataset_name_first, uris_to_gold_labels_expected_first)

        # generate gold labels for the second dataset (the first dataset is kept in a dump file)
        dataset_name_second = self.test_add_and_get_gold_labels_from_dump.__name__ + '_b'
        uris_to_gold_labels_expected_second = generate_random_uris_and_labels(dataset_name_second,
                                                                              num_elements_to_labels,
                                                                              ['Autobots', 'Decepticons'])
        oracle.add_gold_labels(dataset_name_second, uris_to_gold_labels_expected_second)

        # switch back to the first dataset and retrieve data from dump file
        uris_to_retrieve = [uri for uri, labels_dict in uris_to_gold_labels_expected_first]
        uri_to_gold_labels_found = oracle.get_gold_labels(dataset_name_first, uris_to_retrieve)
        self.assert_uri_labels_equal(uris_to_gold_labels_expected_first, uri_to_gold_labels_found)

        # clean both dataset dumps
        loader.clear_gold_labels_file(dataset_name_first)
        loader.clear_gold_labels_file(dataset_name_second)

    def test_get_gold_labels_no_dump_and_no_in_memory(self):
        dataset_name = self.test_add_and_get_gold_labels_no_dump.__name__

        uris_to_retrieve = ['uri1', 'uri2']
        uris_to_gold_labels_expected = []
        uri_to_gold_labels_found = oracle.get_gold_labels(dataset_name, uris_to_retrieve)

        self.assert_uri_labels_equal(uris_to_gold_labels_expected, uri_to_gold_labels_found)
        loader.clear_gold_labels_file(dataset_name)

    def test_sample(self):
        num_elements_to_labels = 1000
        dataset_name = self.test_sample.__name__
        target_category = 'Autobots'
        random_seed = 1
        sample_size = 100

        uris_to_gold_labels_expected = generate_random_uris_and_labels(dataset_name, num_elements_to_labels,
                                                                       [target_category, 'Decepticons'])
        oracle.add_gold_labels(dataset_name, uris_to_gold_labels_expected)
        uris_to_gold = oracle.sample(dataset_name, target_category, sample_size, random_seed)
        self.assertEqual(sample_size, len(uris_to_gold))
        labels_of_sampled_instances = [label for t in uris_to_gold for label in t[1][target_category].labels]
        self.assertIn(LABEL_POSITIVE, labels_of_sampled_instances)
        self.assertIn(LABEL_NEGATIVE, labels_of_sampled_instances)
        loader.clear_gold_labels_file(dataset_name)

    def test_sample_positives(self):
        num_elements_to_labels = 100
        dataset_name = self.test_sample_positives.__name__
        target_category = 'Autobots'
        random_seed = 1
        sample_size = 7

        uris_to_gold_labels_expected = generate_random_uris_and_labels(dataset_name, num_elements_to_labels,
                                                                       [target_category, 'Decepticons'])
        oracle.add_gold_labels(dataset_name, uris_to_gold_labels_expected)
        uris_to_gold_positives_only = oracle.sample_positives(dataset_name, target_category, sample_size, random_seed)
        self.assertEqual(sample_size, len(uris_to_gold_positives_only))
        [self.assertIn(LABEL_POSITIVE, t[1][target_category].labels) for t in uris_to_gold_positives_only]
        loader.clear_gold_labels_file(dataset_name)

    def test_sample_negatives(self):
        num_elements_to_labels = 100
        dataset_name = self.test_sample_negatives.__name__
        target_category = 'Autobots'
        random_seed = 1
        sample_size = 7

        uris_to_gold_labels_expected = generate_random_uris_and_labels(dataset_name, num_elements_to_labels,
                                                                       [target_category, 'Decepticons'])
        oracle.add_gold_labels(dataset_name, uris_to_gold_labels_expected)
        uris_to_gold_negatives_only = oracle.sample_negatives(dataset_name, target_category, sample_size, random_seed)
        self.assertEqual(sample_size, len(uris_to_gold_negatives_only))
        [self.assertIn(LABEL_NEGATIVE, t[1][target_category].labels) for t in uris_to_gold_negatives_only]
        loader.clear_gold_labels_file(dataset_name)


if __name__ == "__main__":
    unittest.main()
