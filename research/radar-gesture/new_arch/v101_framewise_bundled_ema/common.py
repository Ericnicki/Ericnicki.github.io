import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_indices(length, validation_fraction, seed):
    generator = np.random.default_rng(seed)
    indices = generator.permutation(length)
    count = max(1, round(length * validation_fraction))
    return indices[count:].tolist(), indices[:count].tolist()


def make_loaders(dataset, batch_size, workers, validation_fraction, seed):
    fit, validation = random_indices(len(dataset), validation_fraction, seed)
    options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    return (
        DataLoader(Subset(dataset, fit), shuffle=True, drop_last=True, **options),
        DataLoader(Subset(dataset, validation), shuffle=False, **options),
        fit,
        validation,
    )


def append_jsonl(path, value):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(value) + "\n")
