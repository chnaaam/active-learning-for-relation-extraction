import json
import torch
import numpy as np
import random
import os

def fix_torch_seed(random_seed=0):
    torch.manual_seed(random_seed)

    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)

def initialize_directories(config):
    make_directory(config.path.vocab)
    make_directory(config.path.label)
    make_directory(config.path.cache)
    make_directory(config.path.model)
    make_directory(config.path.hub)


def is_existed_directories(path):
    return os.path.isdir(path)

def make_directory(path):
    if not is_existed_directories(path):
        os.mkdir(path)
