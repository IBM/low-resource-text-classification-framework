# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import os
from pathlib import Path

from lrtc_lib.active_learning.active_learning_factory import ActiveLearningFactory
from lrtc_lib.active_learning.strategies import ActiveLearningStrategies, get_compatible_models
from lrtc_lib.train_and_infer_service.train_and_infer_factory import TrainAndInferFactory
from lrtc_lib.training_set_selector.train_and_dev_set_selector_api import TrainingSetSelectionStrategy
from lrtc_lib.train_and_infer_service.model_type import ModelTypes
import lrtc_lib.oracle_data_access.gold_labels_views as glv


def get_project_root_dir():
    path = Path(os.getcwd())
    if "lrtc_lib" in os.listdir(path):  # working directory is the repository directory
        return os.path.join(path, "lrtc_lib")

    # working directory as an inner directory
    requirements_file = 'requirements.txt'
    while requirements_file not in os.listdir(path):
        if path.parent == path:
            raise Exception('project root directory not found')
        path = path.parent
    return path


RESEARCH_MODE = True
ROOT_DIR = get_project_root_dir()


WORKSPACE_DATA_DIR = os.path.normpath(os.path.join(ROOT_DIR, "output/workspaces"))
if not os.path.exists(WORKSPACE_DATA_DIR):
    os.makedirs(WORKSPACE_DATA_DIR)


INFER_CACHE_SIZE = 200000


PROJECT_PROPERTIES = {
    "active_learning_factory": ActiveLearningFactory(),
    "train_and_infer_factory": TrainAndInferFactory(),
    "training_set_selection": TrainingSetSelectionStrategy.ALL_LABELED_PLUS_UNLABELED_AS_NEGATIVE_EQUAL_RATIO,
    "model_type": ModelTypes.NB,
    "active_learning_strategy": ActiveLearningStrategies.HARD_MINING,
    "models_compatible_with_strategies_func": get_compatible_models,
    "data_view_func": glv.single_category,

}
