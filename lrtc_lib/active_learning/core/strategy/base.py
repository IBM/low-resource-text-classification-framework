# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import numpy as np

from lrtc_lib.active_learning.active_learning_api import ActiveLearner
from lrtc_lib.active_learning.strategies import ActiveLearningStrategies


class RandomSampling(ActiveLearner):
    def __init__(self, max_to_consider=10 ** 6):
        self.max_to_consider = max_to_consider

    def get_strategy(self):
        return ActiveLearningStrategies.RANDOM

    def get_recommended_items_for_labeling(self, workspace_id, model_id, dataset_name, category_name, sample_size=1):
        res = self.get_unlabeled_data(workspace_id, dataset_name, category_name, self.max_to_consider)
        scores = self.get_per_element_score(res, workspace_id, model_id, dataset_name, category_name)
        indices = np.sort(np.argpartition(scores, sample_size)[:sample_size])
        res = np.array(res)[indices]
        return res.tolist()

    def get_per_element_score(self, items, workspace_id, model_id, dataset_name, category_name):
        return np.random.rand(len(items))
