# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import pandas as pd
import os
import argparse
import numpy as np


def extract_data(df, dir, label=lambda x, y: None, headers=None):
    results = {}
    for filename in df["file"].unique():
        file_path = os.path.join(dir, filename)
        sub_df = df[df["file"] == filename]
        sort_ids = np.argsort(
            sub_df["id"])  # keep in case keeping exact order is relevant (currently assumed not relevant)
        rev_sort = np.argsort(sort_ids)  # apply that to reverse ordering
        ids = np.array(sub_df["id"].to_list())
        sorted_ids = ids[sort_ids]
        result = []
        if filename.endswith("csv"):
            if headers:
                result = pd.read_csv(file_path, header=None)
                results.columns = headers
            else:
                result = pd.read_csv(file_path)
            result = result.iloc[sorted_ids, :].iloc[rev_sort, :]
        else:
            with open(file_path, encoding="iso-8859-1") as fl:
                for i, line in enumerate(fl):
                    if len(result) < len(sorted_ids) and i == sorted_ids[len(result)]:
                        result.append((label(line, filename), line.rstrip()))
        results[filename] = result
    # if order needs to be retained, iterate over the original dataframe and take each time the next element for the given file
    if filename.endswith("csv"):
        results = pd.concat([res for res in results.values()], axis=0)
    else:
        results = [x  for result in results.values() for x in result]
        headers=["label", "text"]
        results = pd.DataFrame(results, columns=headers)
    return results


if __name__ == '__main__':
    base_dir = ".."
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset")
    args = parser.parse_args()
    ids_dir = os.path.join(base_dir, "ids", args.dataset)
    out_dir = os.path.join(base_dir, "available_datasets", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    raw_dir = os.path.join(base_dir, "raw", args.dataset)
    label_func=lambda x, y: None
    # if "ag_news" in args.dataset:
    #     headers = ["label", "title", "text"]
    if "polarity" in args.dataset:
        label_func = lambda x, y: "positive" if "pos" in y else "negative"
    if "subjec" in args.dataset:
        label_func = lambda x, y: "objective" if "plot" in y else "subjective"
    root, dirs, filenames = next(os.walk(ids_dir))
    for filename in filenames:
        if "_ids" in filename:
            df = pd.read_csv(os.path.join(root, filename))
            res_df = extract_data(df, raw_dir, label=label_func)
            out_file = filename.replace("_ids", "")
            res_df.to_csv(os.path.join(out_dir, out_file))
