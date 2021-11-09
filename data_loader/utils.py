import re
import pickle

pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣]+')

def save_dump(path, data):
    data = pickle.dumps(data)
    
    with open(path, 'wb') as fp:
        pickle.dump(data, fp)

def load_dump(path):
    with open(path, 'rb') as fp:
        data = pickle.load(fp)

    return pickle.loads(data)

def clean_sentence(sentence):
    sentence = pattern.sub(" ", sentence)

    return sentence

