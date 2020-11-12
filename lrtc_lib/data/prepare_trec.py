# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import os
import numpy as np

raw_path = "../raw/trec"
ids_path = "../ids/trec"

with open(os.path.join(raw_path, "train_5500.label"), encoding="iso-8859-1") as origin:
    rows = np.array([line.encode('ascii', 'ignore').decode().strip() for line in origin])
for dataset_part in ["train", "dev"]:  # test is just TREC_10
    with open(f"{ids_path}/{dataset_part}_ids.csv", encoding="iso-8859-1") as ids_fl:
        ids = np.array([int(row.strip()) for row in ids_fl])
    with open(f"trec/{dataset_part}.txt", "w") as out_fl:
        for id in ids:
            out_fl.write(rows[id - 1] + "\torig_line_number\t" + str(id) + "\n")
