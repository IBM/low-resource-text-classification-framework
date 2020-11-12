# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

from lrtc_lib.data_access.core.data_structs import Label


def single_category(category_name: str, uri_categories_and_labels_map: dict):
    """
    keeps a single category out of all available categories
    """
    return {uri: {category_name: cats_labels[category_name]} for uri, cats_labels in
            uri_categories_and_labels_map.items()}


def multi_category_to_single_category_multi_label(new_category_name: str, uri_categories_and_labels_map: dict,
                                                  cat_subset=None):
    """
    performs the following transformation:
    given:
    cat1: true
    cat2: false
    cat3: false
    returns:
    new_cat_name: cat1
    """
    from lrtc_lib.orchestrator.orchestrator_api import LABEL_POSITIVE
    filtered_gold_labels = {}
    for uri, label_dict in uri_categories_and_labels_map.items():
        labels = frozenset([cat for i, cat in enumerate(label_dict) if LABEL_POSITIVE in label_dict[cat].labels])
        filtered_gold_labels[uri] = {new_category_name: Label(labels=labels, metadata={})}
    return filtered_gold_labels
