# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import os
import pandas as pd

"""
Adapted from https://github.com/ewulczyn/wiki-detox/blob/master/src/figshare/Wikipedia%20Talk%20Data%20-%20Getting%20Started.ipynb
"""

input_dir_path = './wiki_attack'
comments_file_path = os.path.join(input_dir_path, 'attack_annotated_comments.tsv')
annotations_file_path = os.path.join(input_dir_path, 'attack_annotations.tsv')


comments = pd.read_csv(comments_file_path, sep='\t', index_col=0)
annotations = pd.read_csv(annotations_file_path,  sep='\t')
# labels a comment as an attack if the majority of annoatators did so
labels = annotations.groupby('rev_id')['attack'].mean() > 0.5
comments.insert(0, 'label', labels)

# remove newline and tab tokens
comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
comments.rename({'comment': 'text'}, axis=1, inplace=True)

train_comments = comments.query("split=='train'")
dev_comments = comments.query("split=='dev'")
test_comments = comments.query("split=='test'")

print(f'train size = {len(train_comments)}')
print(f'dev size = {len(dev_comments)}')
print(f'test size = {len(test_comments)}')

train_comments.to_csv(os.path.join(input_dir_path, 'train.csv'))
dev_comments.to_csv(os.path.join(input_dir_path, 'dev.csv'))
test_comments.to_csv(os.path.join(input_dir_path, 'test.csv'))
