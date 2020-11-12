# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import abc


class BatchScorerApi():

    @abc.abstractmethod
    def compute_batch_score(self, batch):
        raise NotImplementedError("API functions should not be called")

    def get_per_element_score(self, batch):
        raise NotImplementedError("API functions should not be called")
