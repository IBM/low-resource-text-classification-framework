# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

from statistics import mean
from sklearn.neighbors import NearestNeighbors

from lrtc_lib.active_learning.batch_measure_api import BatchScorerApi


class DiversityCalculator(BatchScorerApi):
    def __init__(self, all_data):
        self.all_data = all_data

    def compute_batch_score(self, batch):
        distances = self.get_per_element_score(batch)
        return 1 / mean(distances)

    def get_per_element_score(self, batch):
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(batch)
        distances, indices = nbrs.kneighbors(self.all_data)
        return distances[:, 1]
