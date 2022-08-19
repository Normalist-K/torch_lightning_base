import os
import random
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def load_dataloader(cfg):
    print(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.prepare_data()
    dm.setup(stage=None)
    train_loader = dm.train_dataloader()
    valid_loader = dm.val_dataloader()
    test_loader = dm.predict_dataloader()

    if cfg.trainer.scheduler.name == 'CosineAnnealingLR':
        cfg.len_train_loader = len(train_loader)

    return train_loader, valid_loader, test_loader

def save_submission(cfg, results):
    sub_dict = {'id': [], 'predictions': []}
    for i, label in enumerate(results):
        sub_dict['id'].append(i)
        sub_dict['predictions'].append(label)

    sub_df = pd.DataFrame(sub_dict)
    print(len(sub_df))
    sub_dir = cfg.path.submissions
    sub_name = f'{cfg.name}-{cfg.dt_string}.csv'
    sub_df.to_csv(os.path.join(sub_dir, sub_name), index=False)
    print(f'{sub_name} saved.')