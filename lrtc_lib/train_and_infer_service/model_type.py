# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

class ModelType(object):
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        if isinstance(other, ModelType):
            return self.name == other.name
        else:
            raise TypeError(f"comparing {other.__class__} to ModelType is not allowed! ")

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.name)


class ModelTypes(object):
    NB = ModelType("NB")
    RAND = ModelType("RAND")
    HFBERT = ModelType("HFBERT")

    @classmethod
    def get_all_types(cls):
        return [v for base_class in [cls, *cls.__bases__]
                for k, v in vars(base_class).items() if not callable(getattr(cls, k)) and not k.startswith("__")]


if __name__ == '__main__':
    print([x.name for x in ModelTypes.get_all_types()])
