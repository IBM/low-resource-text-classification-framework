# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import numpy as np

from lrtc_lib.active_learning.strategies import ActiveLearningStrategies
from lrtc_lib.orchestrator import orchestrator_api
from lrtc_lib.active_learning.active_learning_api import ActiveLearner


class RetrospectiveLearner(ActiveLearner):
    def __init__(self, max_to_consider=10 ** 6):
        self.max_to_consider = max_to_consider

    def get_strategy(self):
        return ActiveLearningStrategies.RETROSPECTIVE

    def get_recommended_items_for_labeling(self, workspace_id, model_id, dataset_name, category_name, sample_size=1):
        items = self.get_unlabeled_data(workspace_id, dataset_name, category_name, self.max_to_consider)
        scores = self.get_per_element_score(items, workspace_id, model_id, dataset_name, category_name)
        indices = np.argsort(scores)[::-1]
        indices = indices[:sample_size]
        items = np.array(items)[indices]
        return items.tolist()

    def get_per_element_score(self, items, workspace_id, model_id, dataset_name, category_name):
        return orchestrator_api.infer(workspace_id, category_name, items)["scores"]
