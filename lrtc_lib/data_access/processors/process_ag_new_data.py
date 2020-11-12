# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import os
import pandas as pd

from lrtc_lib.data_access.processors.dataset_part import DatasetPart
from lrtc_lib.data_access.processors.process_csv_data import CsvProcessor


class AgNewsProcessor(CsvProcessor):

    def __init__(self, dataset_part: DatasetPart,dataset_name ="ag_news", imbalanced_postfix=''):
        super().__init__(dataset_name=dataset_name+imbalanced_postfix, dataset_part=dataset_part, context_col='title')

    def _get_all_categories(self):
        test_file = os.path.join(self.RAW_DATA_BASE_DIR, 'ag_news', 'test.csv')
        df = pd.read_csv(test_file, encoding=self.encoding)
        return sorted(df[self.label_col].unique())
