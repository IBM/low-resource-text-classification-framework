# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

from lrtc_lib.training_set_selector.train_and_dev_sets_selectors import TrainAndDevSetsSelectorAllLabeled, TrainAndDevSetsSelectorAllLabeledPlusUnlabeledAsWeakNegative
from lrtc_lib.training_set_selector.train_and_dev_set_selector_api import TrainingSetSelectionStrategy


def get_training_set_selector(selector=TrainingSetSelectionStrategy.ALL_LABELED):
    if selector == TrainingSetSelectionStrategy.ALL_LABELED:
        return TrainAndDevSetsSelectorAllLabeled()
    elif selector == TrainingSetSelectionStrategy.ALL_LABELED_PLUS_ALL_UNLABELED_AS_NEGATIVE:
        return TrainAndDevSetsSelectorAllLabeledPlusUnlabeledAsWeakNegative(negative_ratio=None)  # None means use all unlabeled
    elif selector == TrainingSetSelectionStrategy.ALL_LABELED_PLUS_UNLABELED_AS_NEGATIVE_EQUAL_RATIO:
        return TrainAndDevSetsSelectorAllLabeledPlusUnlabeledAsWeakNegative(negative_ratio=1)
    elif selector == TrainingSetSelectionStrategy.ALL_LABELED_PLUS_UNLABELED_AS_NEGATIVE_X2_RATIO:
        return TrainAndDevSetsSelectorAllLabeledPlusUnlabeledAsWeakNegative(negative_ratio=2)
    elif selector == TrainingSetSelectionStrategy.ALL_LABELED_PLUS_UNLABELED_AS_NEGATIVE_X10_RATIO:
        return TrainAndDevSetsSelectorAllLabeledPlusUnlabeledAsWeakNegative(negative_ratio=10)

    else:
        raise Exception(f"{selector} is not supported")
