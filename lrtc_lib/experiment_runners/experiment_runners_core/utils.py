# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import os

from lrtc_lib.definitions import ROOT_DIR


def get_output_dir():
    path = os.path.join(ROOT_DIR, 'output', 'experiments')
    os.makedirs(path, exist_ok=True)
    return path
