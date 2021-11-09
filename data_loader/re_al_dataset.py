import os
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from .re_data import ReData
from .utils import *

class ReAlDataset(Dataset):

    def __init__(
            self,
            data,
            vocab_path,
            tokenizer,
            model_type,
            marker_type="typed_entity_marker",
            max_len=100):

        super(ReAlDataset, self).__init__()

        self.vocab_path = vocab_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.marker_type = marker_type

        self.token_list = []
        self.label_list = []
        self.PAD_TOKEN = tokenizer.get_pad_token()
        self.PAD_TOKEN_ID = tokenizer.get_pad_token_id()

        self.build(
            re_data=data,
            model_type=model_type)

    def set_train_dataset(self, re_data, model_type):
        self.build(re_data, model_type)

    def build(self, re_data, model_type):
        for data in tqdm(re_data.data, desc=f"Tokenize sentences for {model_type}"):
            try:
                tokens, label = self.tokenizer(data, method=self.marker_type)

                self.token_list.append(tokens)
                self.label_list.append(label)

            except:
                pass

        self.labels, self.l2i, self.i2l = self.create_label_dict(self.label_list)

    def __len__(self):
        return len(self.token_list)

    def __getitem__(self, idx):
        tokens, labels, segments = self.get_input_data(idx)

        token = torch.tensor(tokens)
        label = torch.tensor(labels)
        segments = torch.tensor(segments)

        return token, segments, label

    def create_label_dict(self, label_list=None):
        label_vocab_fn = os.path.join(self.vocab_path, "labels.vocab")
        if os.path.isfile(label_vocab_fn):
            data = load_dump(label_vocab_fn)

            labels = data["labels"]
            l2i = data["l2i"]
            i2l = data["i2l"]

        else:
            labels = list(set(label_list))

            l2i = {l: i for i, l in enumerate(labels)}
            i2l = {i: l for i, l in enumerate(labels)}

            save_dump(path=label_vocab_fn, data={
                "labels": labels,
                "l2i": l2i,
                "i2l": i2l
            })

        return labels, l2i, i2l

    def get_input_data(self, idx):

        token_list = self.token_list[idx]

        tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]"] + token_list)
        segments = [0] * len(tokens)
        labels = self.l2i[self.label_list[idx]]

        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
            segments = segments[:self.max_len]

        elif len(tokens) < self.max_len:
            tokens = tokens + [self.tokenizer.get_pad_token_id()] * (self.max_len - len(tokens))
            segments = segments + [0] * (self.max_len - len(segments))

        return tokens, labels, segments