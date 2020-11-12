# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import ast
import logging
import os
import time
import ujson


def load_cache_from_disk(path_to_model_cache) -> dict:
    if not os.path.isfile(path_to_model_cache):
        return {}
    with open(path_to_model_cache) as reader:
        model_cache = ujson.load(reader)
    model_cache = {ast.literal_eval(x): y for x, y in model_cache.items()}
    return model_cache


def save_cache_to_disk(path_to_model_cache, model_cache):
    os.makedirs(os.path.dirname(path_to_model_cache), exist_ok=True)
    start = time.time()
    with open(path_to_model_cache, "w") as output_file:
        output_file.write(ujson.dumps(model_cache))
    end = time.time()
    logging.info(f"saving {len(model_cache)} items to disk cache took {end-start}")
