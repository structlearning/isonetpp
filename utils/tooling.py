import os
import copy
import json
import yaml
import torch
from collections.abc import Mapping

class ReadOnlyConfig(Mapping):
    def __init__(self, **kwargs):
        self.kv_store = kwargs

    def __getattr__(self, attr):
        return self.__getitem__(attr)

    def __getitem__(self, attr):
        return self.kv_store[attr]

    def __iter__(self):
        return iter(self.kv_store.items())

    def keys(self):
        return self.kv_store.keys()

    def __len__(self):
        return len(self.kv_store)

    def toJSON(self):
        return json.dumps(self.kv_store, default=lambda o: o.__dict__, indent=4)

def seed_everything(seed: int):
    import random, os
    import numpy as np
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def merge_dicts_with_pref(strong, weak):
    for key in weak:
        if key in strong:
            if isinstance(strong[key], dict) and isinstance(weak[key], dict):
                strong[key] = merge_dicts_with_pref(
                    strong=strong[key],
                    weak=weak[key]
                )
            continue
        strong[key] = weak[key]
    return strong

def load_yamls_with_inheritance(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    if 'inherit' in config:
        for parent_relative_path in config.pop('inherit'):
            parent_path = os.path.join(os.path.dirname(config_path), parent_relative_path)
            config = merge_dicts_with_pref(
                strong=config,
                weak=load_yamls_with_inheritance(parent_path)
            )
    return config

def make_read_only(dic):
    result_dic = dic.copy()
    for key, value in result_dic.items():
        if isinstance(value, dict):
            result_dic[key] = make_read_only(value)
    return ReadOnlyConfig(**result_dic)

def read_config(config_path, with_dict=False):
    config = load_yamls_with_inheritance(config_path)
    read_only_config = make_read_only(config)
    if with_dict:
        return read_only_config, config
    else:
        return read_only_config