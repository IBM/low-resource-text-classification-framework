# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import os
import logging
import pandas as pd

from sklearn.metrics import classification_report
from lrtc_lib.experiment_runners.experiment_runners_core.utils import get_output_dir
from lrtc_lib.data_access import data_access_factory
from lrtc_lib.oracle_data_access import oracle_data_access_api
from lrtc_lib.data_access.processors.dataset_part import DatasetPart
from lrtc_lib.orchestrator.orchestrator_api import LABEL_POSITIVE

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

POSITIVE_SAMPLE_SIZE = 10 ** 6
MIN_TERMS_NUM = 3


if __name__ == '__main__':
    datasets_categories_queries = {'trec': {'LOC': ['Where|countr.*|cit.*']}}
    data_access = data_access_factory.get_data_access()
    dataset_part = DatasetPart.DEV
    dataset_part = dataset_part.name.lower()

    results_df = pd.DataFrame(columns=['dataset', 'category', 'query', 'hit_count'])
    for dataset in datasets_categories_queries:
        dataset_for_query = dataset + '_' + dataset_part
        uris = data_access.get_all_text_elements_uris(dataset_for_query)
        gold_labels = oracle_data_access_api.get_gold_labels(dataset_name=dataset_for_query, text_element_uris=uris)
        gold_labels = dict(gold_labels)
        for category in datasets_categories_queries[dataset]:
            for query in datasets_categories_queries[dataset][category]:
                search_res = data_access.sample_text_elements(dataset_name=dataset_for_query,
                                                              sample_size=POSITIVE_SAMPLE_SIZE, query=query)
                query_results = search_res['results']
                hit_count = search_res['hit_count']

                # # filter too short texts
                # query_results = [e for e in query_results if len(e.text.split()) >= MIN_TERMS_NUM]

                logging.info(f'dataset: {dataset_for_query}\tcategory: {category}\t'
                             f'query: {query}\thit_count: {hit_count}')
                logging.debug('\n\t' + '\n\t'.join([t.text for t in query_results]))

                category_gold_labels = [LABEL_POSITIVE in gold_labels[uri][category].labels for uri in uris]
                query_results_uris = [element.uri for element in query_results]
                query_predicted_labels = [uri in query_results_uris for uri in uris]
                cr = classification_report(category_gold_labels, query_predicted_labels, output_dict=True)
                results_dict = {'dataset': dataset, 'category': category, 'query': query, 'hit_count': hit_count,
                                'accuracy': cr['accuracy'], **cr['True']}
                results_df = results_df.append(results_dict, ignore_index=True)
                logging.info(cr['True'])

    results_file_path = os.path.join(get_output_dir(), f'query_baselines_{dataset_part}.csv')
    results_df.to_csv(results_file_path, index=False)
    logging.info(f'results were saved in {results_file_path}')
