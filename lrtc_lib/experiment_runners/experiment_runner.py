# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import abc
import logging
import time
from collections import defaultdict
from typing import List

import numpy as np
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

import lrtc_lib.data_access.data_access_factory as data_access_factory
import lrtc_lib.experiment_runners.experiments_results_handler as res_handler
from lrtc_lib.oracle_data_access import oracle_data_access_api
from lrtc_lib.active_learning.diversity_calculator import DiversityCalculator
from lrtc_lib.active_learning.knn_outlier_calculator import KnnOutlierCalculator
from lrtc_lib.active_learning.strategies import ActiveLearningStrategies
from lrtc_lib.data_access.core.data_structs import TextElement
from lrtc_lib.data_access.data_access_api import DataAccessApi
from lrtc_lib.data_access.data_access_factory import get_data_access
from lrtc_lib.orchestrator import orchestrator_api
from lrtc_lib.orchestrator.orchestrator_api import DeleteModels
from lrtc_lib.train_and_infer_service.model_type import ModelType
from lrtc_lib.training_set_selector.train_and_dev_set_selector_api import TrainingSetSelectionStrategy


@dataclass
class ExperimentParams:
    experiment_name: str
    train_dataset_name: str
    dev_dataset_name: str
    test_dataset_name: str
    category_name: str
    workspace_id: str
    model: ModelType
    active_learning_strategies: list
    repeat_id: int
    train_params: dict


def compute_batch_scores(config, elements):
    data_access = get_data_access()
    unlabeled = data_access.sample_unlabeled_text_elements(config.workspace_id, config.train_dataset_name,
                                                           config.category_name, 10 ** 6)["results"]
    unlabeled_emb = np.array(orchestrator_api.infer(config.workspace_id, config.category_name, unlabeled)["embeddings"])
    batch_emb = np.array(orchestrator_api.infer(config.workspace_id, config.category_name, elements)["embeddings"])

    outlier_calculator = KnnOutlierCalculator(unlabeled_emb)
    outlier_value = outlier_calculator.compute_batch_score(batch_emb)
    representativeness_value = 1 / outlier_value
    diversity_calculator = DiversityCalculator(unlabeled_emb)
    diversity_value = diversity_calculator.compute_batch_score(batch_emb)
    return diversity_value, representativeness_value


class ExperimentRunner(object, metaclass=abc.ABCMeta):
    NO_AL = 'no_active_learning'

    def __init__(self, first_model_positives_num: int, first_model_negatives_num: int,
                 active_learning_suggestions_num: int):
        """
        Init the ExperimentsRunner
        :param first_model_positives_num: the number of positives instances to provide for the first model.
        :param first_model_negatives_num: the number of negative instances to provide for the first model.
        :param active_learning_suggestions_num: the number of instances to be suggested by the active learning strategy
        for the training of the second model.

        """
        self.first_model_positives_num = first_model_positives_num
        self.first_model_negatives_num = first_model_negatives_num
        self.active_learning_suggestions_num = active_learning_suggestions_num
        self.data_access: DataAccessApi = data_access_factory.get_data_access()
        self.cached_first_model_scores = False
        orchestrator_api.set_training_set_selection_strategy(TrainingSetSelectionStrategy.ALL_LABELED)

    def run(self, config: ExperimentParams, active_learning_iterations_num: int, results_file_path: str,
            delete_workspaces: bool = True):

        # key: active learning name, value: list of results oevr iterations (first model has no iterations)
        results_per_active_learning = defaultdict(dict)
        # train first model
        iteration = 0
        res_dict = self.train_first_model(config=config)
        res_handler.save_results(results_file_path, [res_dict])
        results_per_active_learning[self.NO_AL][iteration] = res_dict

        original_workspace_id = config.workspace_id

        for al in config.active_learning_strategies:
            orchestrator_api.set_active_learning_strategy(al)
            if not orchestrator_api.is_model_compatible_with_active_learning(al, config.model):
                logging.info(f'skipping active learning strategy {al.name} for model {config.model.name} '
                             f'since the strategy does not support this model.')
                continue

            al_workspace_id = original_workspace_id + "-" + al.name
            if orchestrator_api.workspace_exists(al_workspace_id):
                orchestrator_api.delete_workspace(al_workspace_id)
            orchestrator_api.copy_workspace(original_workspace_id, al_workspace_id)
            config.workspace_id = al_workspace_id

            for iteration in range(1, active_learning_iterations_num + 1):
                logging.info(f'Run AL strategy: {al.name}, iteration num: {iteration}, repeat num: {config.repeat_id}\t'
                             f'workspace: {config.workspace_id}')

                res_dict, train_id = self.run_active_learning_iteration(config, al, iteration)
                res_handler.save_results(results_file_path, [res_dict])
                results_per_active_learning[al.name][iteration] = res_dict

            if delete_workspaces:
                orchestrator_api.delete_workspace(config.workspace_id, DeleteModels.ALL_BUT_FIRST_MODEL)
        if delete_workspaces:
            orchestrator_api.delete_workspace(original_workspace_id)
        return results_per_active_learning

    def train_first_model(self, config: ExperimentParams):
        if orchestrator_api.workspace_exists(config.workspace_id):
            orchestrator_api.delete_workspace(config.workspace_id)

        orchestrator_api.create_workspace(config.workspace_id, config.train_dataset_name,
                                          dev_dataset_name=config.dev_dataset_name)
        orchestrator_api.create_new_category(config.workspace_id, config.category_name, "No description for you")

        dev_text_elements_uris = orchestrator_api.get_all_text_elements_uris(config.dev_dataset_name)
        dev_text_elements_and_labels = oracle_data_access_api.get_gold_labels(config.dev_dataset_name,
                                                                              dev_text_elements_uris)
        if dev_text_elements_and_labels is not None:
            orchestrator_api.set_labels(config.workspace_id, dev_text_elements_and_labels)

        random_seed = sum([ord(c) for c in config.workspace_id])
        logging.info(str(config))
        logging.info(f'random seed: {random_seed}')

        self.set_first_model_positives(config, random_seed)
        self.set_first_model_negatives(config, random_seed)

        # train first model
        logging.info(f'Starting first model training (model: {config.model.name})\tworkspace: {config.workspace_id}')
        new_model_id = orchestrator_api.train(config.workspace_id, config.category_name, config.model, train_params=config.train_params)
        if new_model_id is None:
            raise Exception(f'a new model was not trained\tworkspace: {config.workspace_id}')

        eval_dataset = config.test_dataset_name
        res_dict = self.evaluate(config, al=self.NO_AL, iteration=0, eval_dataset=eval_dataset)
        res_dict.update(self.generate_al_batch_dict(config))  # ensures AL-related keys are in the results dictionary

        logging.info(f'Evaluation on dataset: {eval_dataset}, iteration: 0, first model (id: {new_model_id}) '
                     f'repeat: {config.repeat_id}, is: {res_dict}\t'
                     f'workspace: {config.workspace_id}')

        return res_dict

    def run_active_learning_iteration(self, config: ExperimentParams, al, iteration):
        # get suggested elements for labeling (and their gold labels)
        suggested_text_elements, suggested_uris_and_gold_labels = \
            self.get_suggested_elements_and_gold_labels(config, al)

        # calculate metrics for the batch suggested by the active learning strategy
        al_batch_dict = self.generate_al_batch_dict(config, suggested_text_elements)

        # set gold labels as the user-provided labels of the elements suggested by the active learning strategy
        orchestrator_api.set_labels(config.workspace_id, suggested_uris_and_gold_labels)

        # train a new model with the additional elements suggested by the active learning strategy
        new_model_id = orchestrator_api.train(config.workspace_id, config.category_name, config.model, train_params=config.train_params)
        if new_model_id is None:
            raise Exception('New model was not trained')

        # evaluate the new model
        eval_dataset = config.test_dataset_name
        res_dict = self.evaluate(config, al.name, iteration, eval_dataset, suggested_text_elements)
        res_dict.update(al_batch_dict)

        logging.info(f'Evaluation on dataset: {eval_dataset}, with AL: {al.name}, iteration: {iteration}, '
                     f'repeat: {config.repeat_id}, model (id: {new_model_id}) is: {res_dict}\t'
                     f'workspace: {config.workspace_id}')
        return res_dict, new_model_id

    def get_suggested_elements_and_gold_labels(self, config, al):
        start = time.time()
        suggested_text_elements_for_labeling = \
            orchestrator_api.get_elements_to_label(config.workspace_id, config.category_name,
                                                   self.active_learning_suggestions_num)
        end = time.time()
        logging.info(f'{len(suggested_text_elements_for_labeling)} instances '
                     f'suggested by active learning strategy: {al.name} '
                     f'for dataset: {config.train_dataset_name} and category: {config.category_name}.\t'
                     f'runtime: {end - start}\tworkspace: {config.workspace_id}')
        uris_for_labeling = [elem.uri for elem in suggested_text_elements_for_labeling]
        uris_and_gold_labels = oracle_data_access_api.get_gold_labels(config.train_dataset_name, uris_for_labeling,
                                                                      config.category_name)
        return suggested_text_elements_for_labeling, uris_and_gold_labels

    def evaluate(self, config: ExperimentParams, al, iteration, eval_dataset,
                 suggested_text_elements_for_labeling=None):
        metadata_dict = res_handler.generate_metadata_dict(config, eval_dataset, al, iteration)
        labels_counts_dict = res_handler.generate_train_labels_counts_dict(config)
        performance_dict = res_handler.generate_performance_metrics_dict(config, eval_dataset)
        experiment_specific_metrics_dict = \
            self.generate_additional_metrics_dict(config, suggested_text_elements_for_labeling)
        res_dict = {**metadata_dict, **labels_counts_dict, **performance_dict, **experiment_specific_metrics_dict}
        return res_dict

    @abc.abstractmethod
    def set_first_model_positives(self, config, random_seed) -> List[TextElement]:
        """
        Set the positive instances for the training of the first model.
        :param config: experiment config for this run
        :param random_seed: a seed for the Random being used for sampling
        :return: a list of TextElements and a log message

        """
        func_name = self.set_first_model_positives.__name__
        raise NotImplementedError('users must define ' + func_name + ' to use this base class')

    @abc.abstractmethod
    def set_first_model_negatives(self, config, random_seed) -> List[TextElement]:
        """
        Set the negative instances for the training of the first model.
        :param config: experiment config for this run
        :param random_seed: a seed for the Random being used for sampling
        :return: a list of TextElements and a log message

        """
        func_name = self.set_first_model_negatives.__name__
        raise NotImplementedError('users must define ' + func_name + ' to use this base class')

    @staticmethod
    def generate_al_batch_dict(config, batch_elements=None):
        batch_dict = {}
        model_supports_embeddings = \
            orchestrator_api.is_model_compatible_with_active_learning(ActiveLearningStrategies.DAL, config.model)
        if batch_elements is not None and model_supports_embeddings:
            diversity_value, representativeness_value = compute_batch_scores(config, batch_elements)
            batch_dict["diversity"] = diversity_value
            batch_dict["representativeness"] = representativeness_value
        else:
            batch_dict["diversity"] = "NA"
            batch_dict["representativeness"] = "NA"
        return batch_dict

    def generate_additional_metrics_dict(self, config, suggested_text_elements_for_labeling):
        return {}
