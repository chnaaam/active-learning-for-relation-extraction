import json

class ReData():
    def __init__(self, path):
        self.data = self.load(path)

    def load(self, path):

        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        return data