# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import os
import pandas as pd
from collections import defaultdict

from lrtc_lib.data_access.core.utils import get_datasets_base_dir
from lrtc_lib.data_access.data_access_in_memory import DataAccessInMemory


def get_all_datasets():
    return sorted(os.listdir(get_datasets_base_dir()))


if __name__ == '__main__':
    from lrtc_lib.oracle_data_access import oracle_data_access_api

    data_access = DataAccessInMemory()
    stats = defaultdict(dict)
    all_datasets = get_all_datasets()
    for dataset in all_datasets:
        dataset_short_name = '_'.join(dataset.split('_')[:-1])
        dataset_part = dataset.split('_')[-1]
        stats[dataset_short_name][dataset_part] = len(data_access.get_all_text_elements_uris(dataset))
        if dataset_part == 'train':
            stats[dataset_short_name]['categories'] = sorted(oracle_data_access_api.get_all_labels(dataset))
    stats_df = pd.DataFrame(stats).transpose()
    pd.set_option('max_columns', None)
    pd.set_option('max_colwidth', 1000)
    print(stats_df)
