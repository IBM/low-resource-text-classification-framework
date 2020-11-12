# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import datetime
import logging
import random
import re
from collections import defaultdict
from typing import Dict, List

import lrtc_lib.experiment_runners.experiments_results_handler as res_handler
from lrtc_lib.experiment_runners.experiment_runner import ExperimentRunner, ExperimentParams
from lrtc_lib.experiment_runners.experiment_runners_core.plot_results import plot_results
from lrtc_lib.oracle_data_access import oracle_data_access_api
from lrtc_lib.active_learning.strategies import ActiveLearningStrategies
from lrtc_lib.data_access.core.data_structs import Label, TextElement
from lrtc_lib.orchestrator import orchestrator_api
from lrtc_lib.orchestrator.orchestrator_api import LABEL_NEGATIVE
from lrtc_lib.train_and_infer_service.model_type import ModelTypes


class ExperimentRunnerImbalancedPractical(ExperimentRunner):
    """
    An experiment in which the set of positive instances for the first model are derived using a query.

    The positive instances for the first model are sampled using a query and are provided with their true gold label
    (i.e., some may actually be negatives).
    The negative instances for the first model are sampled randomly from all other instances, and are set as negatives
    (regardless of their gold label).
    """

    def __init__(self, first_model_labeled_from_query_num: int, first_model_negatives_num: int,
                 active_learning_suggestions_num: int, queries_per_dataset: Dict):
        """
        Init the ExperimentsRunner
        :param first_model_labeled_from_query_num: the number of instances to sample using the query and provide
        for the first model.
        :param first_model_negatives_num: the number of negative instances to provide for the first model.
        :param active_learning_suggestions_num: the number of instances to be suggested by the active learning strategy
        for each iteration (for training the second model and onwards).
        """
        super().__init__(first_model_positives_num=first_model_labeled_from_query_num,
                         first_model_negatives_num=first_model_negatives_num,
                         active_learning_suggestions_num=active_learning_suggestions_num)
        self.queries_per_dataset = queries_per_dataset

    def set_first_model_positives(self, config, random_seed) -> List[TextElement]:
        """
        Choose instances by queries, regardless of their gold label.

        :param config: experiment config for this run
        :param random_seed: a seed for the Random being used for sampling
        :return: a list of TextElements
        """
        general_dataset_name = config.train_dataset_name.split('_train')[0]
        queries = self.queries_per_dataset[general_dataset_name][config.category_name]
        sampled_unlabeled_text_elements = []
        for query in queries:
            sampled_unlabeled_text_elements.extend(
                self.data_access.sample_unlabeled_text_elements(workspace_id=config.workspace_id,
                                                                dataset_name=config.train_dataset_name,
                                                                category_name=config.category_name,
                                                                sample_size=self.first_model_positives_num,
                                                                query=query, remove_duplicates=True)['results']
            )
            logging.info(
                f"Positive sampling, after query {query} size is {len(sampled_unlabeled_text_elements)} ")

        if len(sampled_unlabeled_text_elements) > self.first_model_positives_num:
            random.seed(random_seed)
            sampled_unlabeled_text_elements = random.sample(sampled_unlabeled_text_elements,
                                                            self.first_model_positives_num)

        sampled_uris = [t.uri for t in sampled_unlabeled_text_elements]
        sampled_uris_and_gold_labels = dict(
            oracle_data_access_api.get_gold_labels(config.train_dataset_name, sampled_uris))
        sampled_uris_and_label = \
            [(x.uri, {config.category_name: sampled_uris_and_gold_labels[x.uri][config.category_name]})
             for x in sampled_unlabeled_text_elements]
        orchestrator_api.set_labels(config.workspace_id, sampled_uris_and_label)

        logging.info(f'Set the label of {len(sampled_uris_and_label)} instances sampled by queries {queries} '
                     f'using the oracle for category {config.category_name}')
        logging.info(f"Positive sampling, returned {len(sampled_uris)} elements")

        return sampled_uris

    def set_first_model_negatives(self, config, random_seed) -> List[TextElement]:
        """
         Randomly choose from all unlabeled instances.
        :param config: experiment config for this run
        :param random_seed: a seed for the Random being used for sampling
        :return: a list of TextElements
        """
        sampled_unlabeled_text_elements = \
            self.data_access.sample_unlabeled_text_elements(workspace_id=config.workspace_id,
                                                            dataset_name=config.train_dataset_name,
                                                            category_name=config.category_name,
                                                            sample_size=self.first_model_negatives_num,
                                                            remove_duplicates=True)['results']
        negative_uris_and_label = [(x.uri, {config.category_name: Label(LABEL_NEGATIVE, {})})
                                   for x in sampled_unlabeled_text_elements]
        orchestrator_api.set_labels(config.workspace_id, negative_uris_and_label)

        negative_uris = [x.uri for x in sampled_unlabeled_text_elements]
        logging.info(f'set the label of {len(negative_uris_and_label)} random unlabeled instances as negatives '
                     f'for category {config.category_name}')
        return negative_uris

    def generate_additional_metrics_dict(self, config, suggested_text_elements_for_labeling):
        if suggested_text_elements_for_labeling is not None:
            query_matches = self.count_query_matches_in_elements(config.train_dataset_name, config.category_name,
                                                                 suggested_text_elements_for_labeling)
        else:
            query_matches = "NA"
        query_matches_dict = {'query_matches': query_matches}
        return query_matches_dict

    def count_query_matches_in_elements(self, train_dataset_name, category_name, elements):
        k = 0
        train_dataset_name = '_'.join(train_dataset_name.rsplit('_')[:-1])
        queries = self.queries_per_dataset[train_dataset_name][category_name]
        for element in elements:
            for query in queries:
                if re.match(query, element.text):
                    k += 1
                    continue
        return k


if __name__ == '__main__':
    """
    Queries from Ein-dor et al. 2020:
    * 'ag_news_imbalanced_1': {'1': ["Afghanistan |Albania |Algeria |American Samoa |Andorra |Angola |Anguilla |Antarctica |Antigua and Barbuda |Argentina |Armenia |Aruba |Australia |Austria |Azerbaijan |Bahamas |Bahrain |Bangladesh |Barbados |Belarus |Belgium |Belize |Benin |Bermuda |Bhutan |Bolivia |Bonaire |Bosnia |Botswana |Bouvet Island |Brazil |British Indian Ocean Territory |Brunei |Bulgaria |Burkina Faso |Burundi |Cambodia |Cameroon |Canada |Cape Verde |Cayman Islands |Central African Republic |Chad |Chile |China |Christmas Island |Cocos Islands |Colombia |Comoros |Congo |Congo |Cook Islands |Costa Rica |C�?te d'Ivoire |Croatia |Cuba |Cura�?ao |Cyprus |Czech Republic |Denmark |Djibouti |Dominica |Dominican Republic |Ecuador |Egypt |El Salvador |Equatorial Guinea |Eritrea |Estonia |Ethiopia |Falkland Islands |Faroe Islands |Fiji |Finland |France |French Guiana |French Polynesia |French Southern Territories |Gabon |Gambia |Georgia |Germany |Ghana |Gibraltar |Greece |Greenland |Grenada |Guadeloupe |Guam |Guatemala |Guernsey |Guinea |Guinea-Bissau |Guyana |Haiti |Heard Island and McDonald Islands |Holy See |Honduras |Hong Kong |Hungary |Iceland |India |Indonesia |Iran |Iraq |Ireland |Isle of Man |Israel |Italy |Jamaica |Japan |Jersey |Jordan |Kazakhstan |Kenya |Kiribati |Korea |Kuwait |Kyrgyzstan |Laos |Latvia |Lebanon |Lesotho |Liberia |Libya |Liechtenstein |Lithuania |Luxembourg |Macao |Macedonia |Madagascar |Malawi |Malaysia |Maldives |Mali |Malta |Marshall Islands |Martinique |Mauritania |Mauritius |Mayotte |Mexico |Micronesia |Moldova |Monaco |Mongolia |Montenegro |Montserrat |Morocco |Mozambique |Myanmar |Namibia |Nauru |Nepal |Netherlands |New Caledonia |New Zealand |Nicaragua |Niger |Nigeria |Niue |Norfolk Island |Northern Mariana Islands |Norway |Oman |Pakistan |Palau |Palestine |Panama |Papua New Guinea |Paraguay |Peru |Philippines |Pitcairn |Poland |Portugal |Puerto Rico |Qatar |R�?union |Romania |Russian Federation |Rwanda |Saint Barth�?lemy |Saint Helena |Saint Kitts and Nevis |Saint Lucia |Saint Martin |Saint Pierre and Miquelon |Saint Vincent and the Grenadines |Samoa |San Marino |Sao Tome and Principe |Saudi Arabia |Senegal |Serbia |Seychelles |Sierra Leone |Singapore |Sint Maarten |Slovakia |Slovenia |Solomon Islands |Somalia |South Africa |South Georgia and the South Sandwich Islands |South Sudan |Spain |Sri Lanka |Sudan |Suriname |Svalbard |Swaziland |Sweden |Switzerland |Syria |Taiwan |Tajikistan |Tanzania |Thailand |Timor-Leste |Togo |Tokelau |Tonga |Trinidad and Tobago |Tunisia |Turkey |Turkmenistan |Turks and Caicos |Tuvalu |Uganda |Ukraine |United Arab Emirates |United Kingdom |Uruguay |Uzbekistan |Vanuatu |Venezuela |Vietnam |Virgin Islands |Wallis and Futuna |Western Sahara |Yemen |Zambia |Zimbabwe"]},
    * 'isear': {'fear': ['fear.*|afraid|scared|scary']}
    * 'trec': {'LOC': ['Where|countr.*|cit.*']}
    * 'wiki_attack': {'True': ['[A-Z]!']}
    """

    start_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    # define experiments parameters
    experiment_name = 'query_NB'
    active_learning_iterations_num = 1
    num_experiment_repeats = 1
    # for full list of datasets and categories available run: python -m lrtc_lib.data_access.loaded_datasets_info
    datasets_categories_and_queries = {'trec': {'LOC': ['Where|countr.*|cit.*']}}
    classification_models = [ModelTypes.NB]
    train_params = {ModelTypes.HFBERT: {"metric": "f1"}, ModelTypes.NB: {}}
    active_learning_strategies = [ActiveLearningStrategies.RANDOM, ActiveLearningStrategies.HARD_MINING]

    experiments_runner = ExperimentRunnerImbalancedPractical(first_model_labeled_from_query_num=100,
                                                             first_model_negatives_num=100,
                                                             active_learning_suggestions_num=50,
                                                             queries_per_dataset=datasets_categories_and_queries)

    results_file_path, results_file_path_aggregated = res_handler.get_results_files_paths(
        experiment_name=experiment_name, start_timestamp=start_timestamp, repeats_num=num_experiment_repeats)

    for dataset in datasets_categories_and_queries:
        for category in datasets_categories_and_queries[dataset]:
            for model in classification_models:
                results_all_repeats = defaultdict(lambda: defaultdict(list))
                for repeat in range(1, num_experiment_repeats + 1):
                    config = ExperimentParams(
                        experiment_name=experiment_name,
                        train_dataset_name=dataset + '_train',
                        dev_dataset_name=dataset + '_dev',
                        test_dataset_name=dataset + '_test',
                        category_name=category,
                        workspace_id=f'{experiment_name}-{dataset}-{category}-{model.name}-{repeat}',
                        model=model,
                        active_learning_strategies=active_learning_strategies,
                        repeat_id=repeat,
                        train_params=train_params[model]
                    )

                    # key: active learning name, value: dict with key: iteration number, value: results dict
                    results_per_active_learning = \
                        experiments_runner.run(config,
                                               active_learning_iterations_num=active_learning_iterations_num,
                                               results_file_path=results_file_path,
                                               delete_workspaces=True)
                    for al in results_per_active_learning:
                        for iteration in results_per_active_learning[al]:
                            results_all_repeats[al][iteration].append(results_per_active_learning[al][iteration])

                # aggregate the results of a single active learning iteration over num_experiment_repeats
                if num_experiment_repeats > 1:
                    agg_res_dicts = res_handler.avg_res_dicts(results_all_repeats)
                    res_handler.save_results(results_file_path_aggregated, agg_res_dicts)
    plot_results(results_file_path)
