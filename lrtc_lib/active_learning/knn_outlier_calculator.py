# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from sklearn.neighbors import NearestNeighbors

from lrtc_lib.active_learning.batch_measure_api import BatchScorerApi


class KnnOutlierCalculator(BatchScorerApi):
    def __init__(self,all_data):
        self.all_data = all_data
        self.nbrs = NearestNeighbors(n_neighbors=11, algorithm='ball_tree').fit(self.all_data)

    def get_per_element_score(self, batch):
        distances, indices = self.nbrs.kneighbors(batch)
        return np.mean(distances, axis=-1)

    def compute_batch_score(self, batch):
        return np.mean(self.get_per_element_score(batch))


if __name__ == '__main__':
    all_data = np.random.rand(15000, 50)
    batch = all_data[1:50,:]
    a = KnnOutlierCalculator(all_data)
    print(a.compute_batch_score(batch))