# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import pandas as pd
import os

input_dir_path = './cola/cola_public/raw/'
train_df = pd.read_table(os.path.join(input_dir_path, "in_domain_train.tsv"), header=None)
in_test_df = pd.read_table(os.path.join(input_dir_path, "in_domain_dev.tsv"), header=None)
out_test_df = pd.read_table(os.path.join(input_dir_path, "out_of_domain_dev.tsv"), header=None)
test_df = pd.concat([in_test_df, out_test_df])

train_df = train_df.drop(train_df.columns[[0, 2]], axis=1)
test_df = test_df.drop(test_df.columns[[0, 2]], axis=1)

headers = ["label", "text"]
train_df.columns = headers
test_df.columns = headers

train_df.to_csv("./cola/train.csv", index=None)
test_df.to_csv("./cola/test.csv", index=None)

