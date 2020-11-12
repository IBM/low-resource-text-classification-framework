# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import os
import pandas as pd

from lrtc_lib.data_access.processors.dataset_part import DatasetPart
from lrtc_lib.data_access.processors.process_csv_data import CsvProcessor


class IsearProcessor(CsvProcessor):

    def __init__(self, dataset_part: DatasetPart):
        super().__init__(dataset_name='isear', dataset_part=dataset_part)

    def _get_all_categories(self):
        train_file = os.path.join(self.RAW_DATA_BASE_DIR, 'isear', 'train.csv')
        df = pd.read_csv(train_file)
        return sorted(set(df[self.label_col]))
