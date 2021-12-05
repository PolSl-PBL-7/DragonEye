import json
import pickle


class Json:

    @staticmethod
    def load(path):
        with open(path, 'r') as json_file:
            return json.load(json_file)

    @staticmethod
    def save(dictionary, path):
        with open(path, 'w') as outfile:
            json.dump(dictionary, outfile)


class Pickle:

    @staticmethod
    def load(path):
        with open(path, 'rb') as pickle_file:
            return pickle.load(pickle_file)

    @staticmethod
    def save(object, path):
        with open(path, 'wb') as outfile:
            pickle.dump(object, outfile)
