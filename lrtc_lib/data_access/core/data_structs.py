# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0
from collections import defaultdict
from dataclasses import dataclass

from typing import List, Tuple, Mapping


def nested_default_dict(): return defaultdict(nested_default_dict)


@dataclass
class TextElement:
    uri: str
    text: str
    span: List[Tuple]
    metadata: Mapping
    category_to_label: defaultdict

    @classmethod
    def get_field_names(cls):
        return cls.__annotations__.keys()

@dataclass
class Document:
    uri: str
    text_elements: List[TextElement]
    metadata: Mapping


@dataclass
class Label:
    labels: frozenset
    metadata: Mapping

    def __init__(self, labels, metadata: Mapping):
        if type(labels) == str:
            self.labels = frozenset([labels])
        elif type(labels) == list:
            self.labels = frozenset(labels)
        else:
            self.labels = labels
        self.metadata = metadata

    def to_dict(self):
        dict_for_json = {'labels': list(self.labels), 'metadata': self.metadata}
        return dict_for_json


