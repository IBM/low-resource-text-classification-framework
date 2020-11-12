# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

from lrtc_lib.data_access.processors.dataset_part import DatasetPart
from lrtc_lib.data_access.processors.process_csv_data import CsvProcessor


class ColaProcessor(CsvProcessor):

    def __init__(self, dataset_part: DatasetPart, imbalanced_postfix=''):
        super().__init__(dataset_name='cola'+imbalanced_postfix, dataset_part=dataset_part)

    def _get_all_categories(self):
        return [0, 1]
