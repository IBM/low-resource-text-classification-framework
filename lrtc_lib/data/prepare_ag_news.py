# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import html
import os
import re

import pandas as pd

input_dir_path = "./ag_news"


def clean_text(x):
    x = re.sub('#\S+;', '&\g<0>', x)
    x = re.sub('(\w+)\\\(\w+)', '\g<1> \g<2>', x)
    x = x.replace('quot;', '&quot;')
    x = x.replace('amp;', '&amp;')
    x = x.replace('\$', '$')
    while x.endswith("\\"):
        x = x[:-1]
    return html.unescape(x)


for dataset_part in ["train", "test"]:
    dataset_part = dataset_part.lower()
    txt_input = os.path.join(input_dir_path, f'{dataset_part}.csv')
    print("looking for", os.path.abspath(txt_input))
    csv_output = os.path.join(input_dir_path, f'{dataset_part}_orig.csv')

    df = pd.read_csv(txt_input, header=None)
    df.columns = ["label", "title", "text"]

    df.title = df.title.apply(lambda x: clean_text(x))
    df.text = df.text.apply(lambda x: clean_text(x))
    df.to_csv(csv_output, index=False)
    print(f'converted {len(df)} lines from {txt_input}')
