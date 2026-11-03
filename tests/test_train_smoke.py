"""Smoke test: 2 epochs on fake data (2 blocks to keep it fast), checkpoints + log + lr schedule."""

import csv
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from data.synthetic import make_synthetic_eyepacs
from train.train import build_scheduler, main
from train.utils import load_checkpoint
from vim import build_model
from vim.config import config_from_dict

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")


def test_two_epoch_training_run(tmp_path):
    #50 patients x 2 eyes = 100 images
    image_dir, labels_csv = make_synthetic_eyepacs(str(tmp_path / "data"),
                                                   num_patients=50)
    out_dir = tmp_path / "run"

    #same as typing it on the command line, forward slashes for yaml
    argv = [
        "--config", CONFIG_PATH,
        "--smoke-test",
        "--device", "cpu",
        "--override", f"data.eyepacs_image_dir={image_dir.replace(os.sep, '/')}",
        "--override", f"data.eyepacs_labels_csv={labels_csv.replace(os.sep, '/')}",
        "--override", f"train.output_dir={out_dir.as_posix()}",
        "--override", "model.depth=2",
    ]
    summary = main(argv)

    #smoke test adds _smoke to the folder
    out_dir = tmp_path / "run_smoke"
    assert summary["output_dir"].replace(os.sep, "/") == out_dir.as_posix()

    assert os.path.exists(out_dir / "best.pt")
    assert os.path.exists(out_dir / "last.pt")
    assert os.path.exists(out_dir / "log.csv")
    assert os.path.exists(out_dir / "config.yaml")

    #2 rows, finite losses
    with open(out_dir / "log.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print("Log rows:", rows)
    assert len(rows) == 2
    for row in rows:
        assert math.isfinite(float(row["train_loss"]))
        assert math.isfinite(float(row["val_loss"]))

    #best.pt loads back using the config saved inside it
    state = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=True)
    cfg = config_from_dict(state["config"])
    model = build_model(cfg.model)
    load_checkpoint(str(out_dir / "best.pt"), model)
    assert cfg.model.depth == 2
    assert summary["best_qwk"] > float("-inf")


def test_scheduler_warms_up_then_decays():
    #dummy param just to drive the schedule
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    scheduler = build_scheduler(optimizer, base_lr=1e-3, min_lr=1e-5,
                                warmup_steps=10, total_steps=100)

    lrs = []
    for _ in range(100):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    #warmup up to the peak
    assert lrs[0] < 2e-4
    assert abs(lrs[9] - 1e-3) < 1e-9
    #then only goes down
    for step in range(9, 99):
        assert lrs[step] >= lrs[step + 1]
    #ends near min_lr
    assert lrs[-1] < 2e-5


if __name__ == "__main__":
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        test_two_epoch_training_run(pathlib.Path(folder))
    test_scheduler_warms_up_then_decays()
    print("OK")
