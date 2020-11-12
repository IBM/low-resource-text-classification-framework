# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import logging
import pickle
import shutil
import uuid
import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB

from lrtc_lib.definitions import ROOT_DIR
from lrtc_lib.train_and_infer_service.train_and_infer_api import ModelStatus, TrainAndInferAPI, infer_with_cache


class TrainAndInferNB(TrainAndInferAPI):
    def __init__(self, max_datapoints=10000,
                 model_relative_path=os.path.join("output", "models", "nb")):
        super(TrainAndInferNB, self).__init__()
        model_absolute_path = os.path.join(ROOT_DIR, model_relative_path)
        if not os.path.isdir(model_absolute_path):
            os.makedirs(model_absolute_path)
        self.model_relative_path = model_relative_path
        self.model = MultinomialNB()
        self.vocab = {}
        self.rev_vocab = {}
        self.features_num = 0
        self.max_datapoints = max_datapoints
        self.infer_batch_size = max_datapoints
        self.model_id = -1

    def normalize(self, word):
        return word.lower()

    def create_vocab(self, sentences):
        words = set()
        for sentence in sentences:
            for word in sentence.split():
                words.add(self.normalize(word))
        self.vocab = {i: word for i, word in enumerate(words)}
        self.rev_vocab = {word: i for i, word in self.vocab.items()}
        self.features_num = len(words) + 1

    def input_to_features(self, input_data):
        if len(input_data) > self.max_datapoints:
            logging.info(f"Too many datapoints for NB ({len(input_data)}), using only {self.max_datapoints}")
            data_len = self.max_datapoints
        else:
            data_len = len(input_data)
        features = np.zeros((data_len, self.features_num))
        for sentence_num, sentence in zip(range(data_len), input_data):
            for word in sentence.split():
                if self.normalize(word) in self.rev_vocab:
                    feature_num = self.rev_vocab[self.normalize(word)]
                else:
                    feature_num = self.features_num - 1
                features[sentence_num, feature_num] += 1
        return features

    def train(self, train_data, dev_data, test_data, train_params):
        self.model_id = str(uuid.uuid1())
        train_file = self.train_file_by_id(self.model_id)
        model_file = self.model_file_by_id(self.model_id)
        params_file = self.params_file_by_id(self.model_id)
        try:
            with open(train_file, "w") as fl:
                fl.write("")

            # Train the model using the training sets
            sentences = [sentence["text"] for sentence in train_data]
            sentences = sentences[:self.max_datapoints]
            labels = [sentence["label"] for sentence in train_data]
            labels = labels[:self.max_datapoints]
            self.create_vocab(sentences=sentences)
            features = self.input_to_features(sentences)
            self.model.fit(features, labels)

            os.makedirs(self.get_model_dir_by_id(self.model_id))
            with open(params_file, "wb") as fl:
                pickle.dump(self, fl)
            with open(model_file, "wb") as fl:
                pickle.dump(self.model, fl)

        except Exception as e:
            self.delete_model(self.model_id)
            raise e

        finally:
            if os.path.isfile(train_file):
                os.remove(train_file)

        return self.model_id

    @infer_with_cache
    def infer(self, model_id, items_to_infer, infer_params=None, use_cache=True):
        with open(self.params_file_by_id(model_id), "rb") as fl:
            self = pickle.load(fl)
        with open(self.model_file_by_id(model_id), "rb") as fl:
            self.model = pickle.load(fl)
        items_to_infer = [x["text"] for x in items_to_infer]
        last_batch = 0
        predicted = []
        while last_batch < len(items_to_infer):
            batch = items_to_infer[last_batch:last_batch + self.infer_batch_size]
            last_batch += self.infer_batch_size
            batch = self.input_to_features(batch)
            predicted.append(self.model.predict_proba(batch))
        predicted = np.concatenate(predicted, axis=0)

        labels = [int(np.argmax(prediction)) for prediction in predicted]
        scores = [prediction[1] for prediction in predicted]
        return {"labels": labels, "scores": scores}

    def model_file_by_id(self, model_id):
        return os.path.join(self.get_model_dir_by_id(model_id), "nb_model")

    def params_file_by_id(self, model_id):
        return os.path.join(self.get_model_dir_by_id(model_id), "nb_params")

    def get_models_dir(self):
        return os.path.join(ROOT_DIR, self.model_relative_path)

    def get_model_dir_by_id(self, model_id):
        return os.path.join(self.get_models_dir(), model_id)

    def train_file_by_id(self, model_id):
        return os.path.join(ROOT_DIR, self.model_relative_path, "nb_training_" + model_id)

    def get_model_status(self, model_id):
        if os.path.isfile(self.train_file_by_id(model_id)):
            if not os.path.isdir((self.get_model_dir_by_id(model_id))):
                return ModelStatus.TRAINING
            else:
                return ModelStatus.READY
        elif os.path.isdir((self.get_model_dir_by_id(model_id))):
            return ModelStatus.READY
        return ModelStatus.ERROR

    def delete_model(self, model_id):
        train_file = self.train_file_by_id(model_id)
        model_file = self.model_file_by_id(model_id)
        params_file = self.params_file_by_id(model_id)
        model_dir = self.get_model_dir_by_id(model_id)
        if os.path.isfile(train_file):
            os.remove(train_file)
        if os.path.isfile(model_file):
            os.remove(model_file)
        if os.path.isfile(params_file):
            os.remove(params_file)
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir)


if __name__ == '__main__':
    nb = TrainAndInferNB()
