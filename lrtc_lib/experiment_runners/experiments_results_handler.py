# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import csv
import os

from lrtc_lib.experiment_runners.experiment_runners_core.assessment.evaluate_predictions import evaluate_predictions
from lrtc_lib.experiment_runners.experiment_runners_core.utils import get_output_dir
from lrtc_lib.oracle_data_access import oracle_data_access_api
from lrtc_lib.orchestrator import orchestrator_api

TRAIN_NEGATIVE_COUNT_HEADER = "train negative count"
TRAIN_POSITIVE_COUNT_HEADER = "train positive count"
TRAIN_TOTAL_COUNT_HEADER = "train total count"
ITERATION_HEADER = 'iteration number'
REPEAT_HEADER = 'repeat id'


def generate_metadata_dict(config, eval_dataset, al, iteration_num):
    res_dict = {'dataset': eval_dataset, 'category': config.category_name, 'model': config.model.name, 'AL': al,
                ITERATION_HEADER: iteration_num, REPEAT_HEADER: config.repeat_id}
    return res_dict


def generate_train_labels_counts_dict(config):
    counts_dict = {}
    counts = orchestrator_api.get_label_counts(config.workspace_id, config.train_dataset_name, config.category_name)
    counts_dict[TRAIN_POSITIVE_COUNT_HEADER] = counts["true"]
    counts_dict[TRAIN_NEGATIVE_COUNT_HEADER] = counts["false"]
    counts_dict[TRAIN_TOTAL_COUNT_HEADER] = counts["true"] + counts["false"]
    return counts_dict


def generate_performance_metrics_dict(config, evaluation_dataset):
    all_text_elements = orchestrator_api.get_all_text_elements(evaluation_dataset)
    all_text_elements_uris = [elem.uri for elem in all_text_elements]

    predicted_labels = orchestrator_api.infer(config.workspace_id, config.category_name, all_text_elements)
    uris_and_gold_labels = oracle_data_access_api.get_gold_labels(evaluation_dataset, all_text_elements_uris)
    gold_of_category = [x[1][config.category_name].labels for x in uris_and_gold_labels]
    performance_metrics = evaluate_predictions(gold_of_category, predicted_labels)
    return performance_metrics


def save_results(res_file_name: str, res_dicts: list):
    header = res_dicts[0].keys()
    if len(header) == 0:
        return
    first_write_to_res_file = not os.path.exists(res_file_name)
    with open(res_file_name, 'a', newline='') as f:
        w = csv.DictWriter(f, header)
        if first_write_to_res_file:
            w.writeheader()
        w.writerows(res_dicts)


def get_results_files_paths(experiment_name, start_timestamp, repeats_num=None, prefix=None):
    res_dir = os.path.join(get_output_dir(), experiment_name + '_' + start_timestamp, "results")
    os.makedirs(res_dir, exist_ok=True)

    experiment_name = experiment_name if prefix is None else prefix + '_' + experiment_name
    results_file_path_all_experiment_repeats = os.path.join(res_dir, f'{experiment_name}_all_repeats.csv')

    results_file_path_aggregated = None
    if repeats_num is not None and repeats_num > 1:
        results_file_path_aggregated = os.path.join(res_dir, f'{experiment_name}_{repeats_num}_repeats_avg.csv')

    return results_file_path_all_experiment_repeats, results_file_path_aggregated


def avg_res_dicts(results_all_repeats):
    avg_postfix = '_avg'
    avg_dict_per_al_iter = []
    for al in results_all_repeats:
        for iteration in results_all_repeats[al]:
            avg_dict = {}
            for key in results_all_repeats[al][iteration][0].keys():
                if isinstance(results_all_repeats[al][iteration][0][key], float) or \
                        isinstance(results_all_repeats[al][iteration][0][key], int):
                    avg_dict[key + avg_postfix] = \
                        sum([res_dict[key] for res_dict in results_all_repeats[al][iteration]]) / \
                        float(len(results_all_repeats[al][iteration]))
                else:
                    avg_dict[key + avg_postfix] = results_all_repeats[al][iteration][0][key]
            avg_dict_per_al_iter.append(avg_dict)
    return avg_dict_per_al_iter
