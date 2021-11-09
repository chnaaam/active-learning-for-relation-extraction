import copy
import os

from . import ReData
from . import ReAlDataset
from . import ReAlDataModule


class ReDataManager:
    def __init__(self, config, model_type, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        self.train_data = ReData(os.path.join(config.path.data, config.dataset.train))
        self.train_data_pool = copy.deepcopy(self.train_data)
        self.total_data_length = len(self.train_data.data)
        self.data_interval = self.total_data_length // 10

        self.data_module = ReAlDataModule(config, model_type, tokenizer)

        self.seed_idxes = [i for i in range(self.data_interval)]

    def get_dataset(self, data_idx_list):
        data = []

        for data_idx in data_idx_list:
            data.append(self.train_data_pool.data[data_idx])

        return data

    def set_idx_list(self, idx_list):
        self.seed_idxes += idx_list

    def get_unlabeled_data(self):
        unlabeled_idx_list = []
        for i in range(self.total_data_length):
            if i not in self.seed_idxes:
                unlabeled_idx_list.append(i)

        return [(i, self.train_data_pool.data[i]) for i in unlabeled_idx_list]

    def set_data_module(self, data_idx):
        self.train_data.data = [self.train_data_pool.data[idx] for idx in data_idx]

        self.data_module.change_train_dataset(dataset=ReAlDataset(
            data=self.train_data,
            vocab_path=self.config.path.vocab,
            tokenizer=self.tokenizer,
            model_type=self.config.tokenizer.marker_type
        ))