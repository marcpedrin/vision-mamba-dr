"""
Training helpers: seeding, running average, checkpoints, CSV log.

    Everything seeded so the same config + seed gives the same run
    (on the same hardware).

"""

import csv
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy, torch cpu and torch gpu."""
    #for child processes like dataloader workers
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    #weights init, shuffling, augmentations
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #make cuDNN pick the same algorithm every time (bit slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed numpy/python inside each dataloader worker (torch already does its own)."""
    #different per worker but deterministic, numpy needs < 2**32
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class AverageMeter:
    #running average weighted by batch size

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1) -> None:
        #value is the mean of n items
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(path: str, model, optimizer=None, scheduler=None, **extra) -> None:
    """Save weights (+ optimizer/scheduler state and any extra named values)."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    #name -> tensor
    state = {"model": model.state_dict()}

    #so a run can resume where it stopped
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()

    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()

    #epoch, metrics, config...
    state.update(extra)

    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, device="cpu"):
    """Load a checkpoint into model, returns the whole dict."""
    #weights_only=True so a bad file cant run code
    state = torch.load(path, map_location=device, weights_only=True)

    #strict, errors if the config doesnt match
    model.load_state_dict(state["model"])

    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])

    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])

    return state


class Logger:
    #one csv row per epoch, eval/plots.py reads it for the curves

    def __init__(self, path: str):
        self.path = path
        #columns come from the first row logged
        self.fieldnames = None

        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        #empty the file for a fresh run
        with open(self.path, "w", newline="", encoding="utf-8"):
            pass

    def log(self, row: dict) -> None:
        first_row = self.fieldnames is None
        if first_row:
            self.fieldnames = list(row.keys())

        #append so old rows stay
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if first_row:
                writer.writeheader()
            writer.writerow(row)
