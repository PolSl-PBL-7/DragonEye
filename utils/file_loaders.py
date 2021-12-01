import json

class Json:

    @staticmethod
    def load(path):
        with open(path) as json_file:
            return json.load(json_file)
    
    @staticmethod
    def save(dictionary, path):
        with open(path) as outfile:
            json.dump(dictionary, outfile)