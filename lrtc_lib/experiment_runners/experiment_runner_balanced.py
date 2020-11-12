# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import datetime
import logging
from collections import defaultdict
from typing import List

import lrtc_lib.experiment_runners.experiments_results_handler as res_handler
from lrtc_lib.experiment_runners.experiment_runner import ExperimentRunner, ExperimentParams
from lrtc_lib.oracle_data_access import oracle_data_access_api
from lrtc_lib.active_learning.strategies import ActiveLearningStrategies
from lrtc_lib.data_access.core.data_structs import TextElement
from lrtc_lib.orchestrator import orchestrator_api
from lrtc_lib.experiment_runners.experiment_runners_core.plot_results import plot_results
from lrtc_lib.train_and_infer_service.model_type import ModelTypes


class ExperimentRunnerBalanced(ExperimentRunner):
    """
    An experiment over balanced data.

    The positive instances for the first model are sampled randomly from the true positive instances.
    The negative instances for the first model are sampled randomly from all other instances, and are set as negatives
    (regardless of their gold label).
    """

    def __init__(self, first_model_labeled_num: int, active_learning_suggestions_num: int):
        """
        Init the ExperimentRunner
        :param first_model_labeled_num: the number of labeled instances to provide for the first model.
        :param active_learning_suggestions_num: the number of instances to be suggested by the active learning strategy
        for each iteration (for training the second model and onwards).
        """
        super().__init__(first_model_positives_num=first_model_labeled_num, first_model_negatives_num=0,
                         active_learning_suggestions_num=active_learning_suggestions_num)

    def set_first_model_positives(self, config, random_seed) -> List[TextElement]:
        """
        Randomly choose instances, regardless of their gold label.
        :param config: experiment config for this run
        :param random_seed: a seed for the Random being used for sampling
        :return: a list of TextElements
        """
        sample_size = self.first_model_positives_num
        sampled_elements = self.data_access.sample_text_elements(config.train_dataset_name, sample_size,
                                                                 remove_duplicates=True)['results']
        sampled_uris = [element.uri for element in sampled_elements]
        sampled_uris_with_labels = \
            oracle_data_access_api.get_gold_labels(config.train_dataset_name, sampled_uris, config.category_name)
        orchestrator_api.set_labels(config.workspace_id, sampled_uris_with_labels)

        logging.info(f'set the label of {len(sampled_uris_with_labels)} random instances with their gold label '
                     f'(can be positive or negative) for category {config.category_name}')
        return sampled_uris

    def set_first_model_negatives(self, config, random_seed) -> List[TextElement]:
        """
        No need for this function in balanced datasets - do nothing.
        :param config: experiment config for this run
        :param random_seed: a seed for the Random being used for sampling
        :return: an empty list
        """

        logging.info('No need for setting negatives in balanced datasets - return an empty list')
        return []


if __name__ == '__main__':
    start_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    # define experiments parameters
    experiment_name = 'balanced_NB'
    active_learning_iterations_num = 5
    num_experiment_repeats = 2
    # for full list of datasets and categories available run: python -m lrtc_lib.data_access.loaded_datasets_info
    datasets_and_categories = {'trec': ['LOC']}
    classification_models = [ModelTypes.NB]
    train_params = {ModelTypes.HFBERT: {"metric": "accuracy"}, ModelTypes.NB: {}}
    active_learning_strategies = [ActiveLearningStrategies.RANDOM, ActiveLearningStrategies.HARD_MINING]

    experiments_runner = ExperimentRunnerBalanced(first_model_labeled_num=100,
                                                  active_learning_suggestions_num=50)

    results_file_path, results_file_path_aggregated = res_handler.get_results_files_paths(
        experiment_name=experiment_name, start_timestamp=start_timestamp, repeats_num=num_experiment_repeats)

    for dataset in datasets_and_categories:
        for category in datasets_and_categories[dataset]:
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
