# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

from typing import Dict, FrozenSet, Sequence

from sklearn.metrics import classification_report

from lrtc_lib.orchestrator.orchestrator_api import LABEL_POSITIVE, BINARY_LABELS


def evaluate_predictions(gold_labels: Sequence[FrozenSet[str]], labels_and_scores: Dict[str, Sequence],
                         binary_positive_label=None):
    """
    :param gold_labels:
    :param labels_and_scores:
    :rtype: dict with all the relevant measurements (Precision, Recall etc...)
    """
    results_dict = {}
    if "scores" in labels_and_scores:
        score = labels_and_scores['scores']
        results_dict['average_score'] = sum(score) / len(score)

    predicted_label = labels_and_scores['labels']
    prediction_correct = [y in x for x, y in zip(gold_labels, predicted_label)]
    results_dict['accuracy'] = 0 if len(prediction_correct) == 0 else sum(prediction_correct) / len(prediction_correct)

    def extract_label(labels):
        if len(labels) == 1:
            return next(iter(labels))
        else:
            raise ValueError("Multi label is not supported")

    gold_labels = [extract_label(l) for l in gold_labels]
    cr = classification_report(gold_labels, predicted_label, output_dict=True)

    is_binary_label = binary_positive_label is not None or set(gold_labels).issubset(BINARY_LABELS)
    if is_binary_label:
        binary_positive_label = binary_positive_label or LABEL_POSITIVE
        results_dict['precision'] = cr[binary_positive_label]['precision']
        results_dict['recall'] = cr[binary_positive_label]['recall']
        results_dict['f1'] = cr[binary_positive_label]['f1-score']
        results_dict['support'] = cr[binary_positive_label]['support']
        numeric_gold_labels = [1 if binary_positive_label in l else 0 for l in gold_labels]
        results_dict['tp'] = sum([y * x for x, y in zip(prediction_correct, numeric_gold_labels)])
        results_dict['fp'] = sum([(1 - x) * (1 - y) for x, y in zip(prediction_correct, numeric_gold_labels)])
        results_dict['tn'] = sum([x * (1 - y) for x, y in zip(prediction_correct, numeric_gold_labels)])
        results_dict['fn'] = sum([(1 - x) * y for x, y in zip(prediction_correct, numeric_gold_labels)])
    else:
        results_dict['precision'] = cr['weighted avg']['precision']
        results_dict['recall'] = cr['weighted avg']['recall']
        results_dict['f1'] = cr['weighted avg']['f1-score']
    return results_dict


if __name__ == '__main__':
    gold_labels = ["true", "true", "false", "true", "false", "false", "true", "true", "true", "false", "false", "false"]
    gold_labels = [frozenset([l]) for l in gold_labels]
    predicted_labels = {
        'labels': ["true", "false", "true", "false", "false", "false", "true", "false", "false", "false", "false",
                   "false"],
        'scores': [0.9, 0.9, 0.8, 0.1, 0.2, 0.3, 1, 1, 1, 1, 1, 1]}
    evaluated_preds = evaluate_predictions(gold_labels, predicted_labels, "true")
    print(evaluated_preds)
