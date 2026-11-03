"""
Train Vision Mamba on EyePACS. Run from the repo root:

    python -m train.train --config configs/default.yaml
    python -m train.train --override train.epochs=5
    python -m train.train --smoke-test       (2 epochs, 100 images, any laptop)

    Writes best.pt (best val QWK), last.pt, log.csv and config.yaml to train.output_dir.
    Paper Sec 4.1: AdamW, wd 0.05, lr 1e-3, cosine, 300 epochs, batch 1024.
    Ours: 30 epochs, batch 32, no mixup/erasing/crops/EMA, class weighted loss.

"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import yaml

from data import build_dataloaders
from data.synthetic import make_synthetic_eyepacs
from eval.metrics import compute_all_metrics
from vim import build_model, count_parameters
from vim.config import config_to_dict, load_config

from .engine import train_one_epoch, validate
from .utils import Logger, save_checkpoint, set_seed


def parse_args(argv=None):
    """Command line options (tests pass a list instead)."""
    parser = argparse.ArgumentParser(description="Train Vision Mamba on EyePACS")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="path to the YAML config")
    #can be given more than once
    parser.add_argument("--override", action="append", default=[],
                        help="change one config value, e.g. train.epochs=5")
    parser.add_argument("--smoke-test", action="store_true",
                        help="2 epochs on 100 images, to check the pipeline end to end")
    parser.add_argument("--device", default="auto", help="'auto', 'cuda' or 'cpu'")
    return parser.parse_args(argv)


def class_weights_from_labels(labels, num_classes):
    """Inverse frequency weights N / (K * count_c), rare grades count more.

    EyePACS is ~73% grade 0, so without this the model just predicts 0.
    """
    counts = torch.zeros(num_classes)
    for label in labels:
        counts[label] += 1
    #missing class in tiny smoke data -> avoid divide by zero
    counts = counts.clamp(min=1)
    weights = counts.sum() / (num_classes * counts)
    return weights


def build_optimizer(model, lr, weight_decay):
    """AdamW, no weight decay on biases, norms, pos_embed, cls_token and A_log."""
    #decaying A_log would pull every A towards -1 and lose the spread of memory lengths
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        #1D = biases and LayerNorm
        if param.ndim <= 1 or name.endswith(("pos_embed", "cls_token", "A_log")):
            no_decay.append(param)
        else:
            decay.append(param)
    #0.9 = paper's momentum
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.999),
    )


def build_scheduler(optimizer, base_lr, min_lr, warmup_steps, total_steps):
    """Linear warmup then cosine down to min_lr, stepped every batch."""

    #LambdaLR multiplies base_lr by this, so return lr / base_lr
    def lr_factor(step):
        #warmup: ~0 -> base_lr
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        #0 -> 1 over the rest of training
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = min_lr + (base_lr - min_lr) * cosine
        return lr / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)


def prepare_smoke_test(cfg):
    """Shrink the config for a quick check, makes fake images if no EyePACS csv."""
    cfg.train.epochs = cfg.train.smoke_test_epochs
    cfg.train.batch_size = cfg.train.smoke_test_batch_size
    #separate folder so it never overwrites a real run
    cfg.train.output_dir = cfg.train.output_dir + "_smoke"
    cfg.train.warmup_epochs = min(cfg.train.warmup_epochs, 1)
    cfg.data.num_workers = 0

    if not os.path.exists(cfg.data.eyepacs_labels_csv):
        #50 patients x 2 eyes = 100 images
        folder = os.path.join(cfg.train.output_dir, "synthetic_data")
        num_patients = max(cfg.train.smoke_test_images // 2, 5)
        print(f"[smoke test] {cfg.data.eyepacs_labels_csv} not found; writing "
              f"{2 * num_patients} SYNTHETIC images to {folder}")
        image_dir, labels_csv = make_synthetic_eyepacs(folder, num_patients)
        cfg.data.eyepacs_image_dir = image_dir
        cfg.data.eyepacs_labels_csv = labels_csv
    return cfg


def main(argv=None):
    """Full training run. Returns a small summary (used by tests)."""
    args = parse_args(argv)

    cfg = load_config(args.config, args.override)
    if args.smoke_test:
        cfg = prepare_smoke_test(cfg)
    max_images = cfg.train.smoke_test_images if args.smoke_test else None

    #seed before anything random happens
    set_seed(cfg.seed)

    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)
    #amp only on gpu
    use_amp = cfg.train.amp and device.type == "cuda"
    print(f"Device: {device}  |  mixed precision: {use_amp}")

    #save a copy of the exact config used
    out_dir = cfg.train.output_dir
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config_to_dict(cfg), f, sort_keys=False)

    #test loader not used here, only in eval/evaluate.py at the end
    train_loader, val_loader, _ = build_dataloaders(cfg, max_images=max_images)
    print(f"Images: train {len(train_loader.dataset)}, "
          f"val {len(val_loader.dataset)}")
    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        raise ValueError(
            "The train or validation split is empty: use more images "
            "(train.smoke_test_images) or a larger data.val_fraction"
        )

    model = build_model(cfg.model).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    #cross entropy, weighted by class frequency
    if cfg.train.class_weights:
        weights = class_weights_from_labels(train_loader.dataset.labels,
                                            cfg.model.num_classes)
        print("Class weights:", [round(w, 2) for w in weights.tolist()])
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(model, cfg.train.lr, cfg.train.weight_decay)
    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(
        optimizer,
        base_lr=cfg.train.lr,
        min_lr=cfg.train.min_lr,
        warmup_steps=cfg.train.warmup_epochs * steps_per_epoch,
        total_steps=cfg.train.epochs * steps_per_epoch,
    )
    #does nothing when amp is off
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    logger = Logger(os.path.join(out_dir, "log.csv"))
    #so the first epoch always counts as best
    best_qwk = float("-inf")
    best_path = os.path.join(out_dir, "best.pt")
    last_path = os.path.join(out_dir, "last.pt")

    for epoch in range(1, cfg.train.epochs + 1):
        start = time.time()

        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, scaler,
            use_amp=use_amp, grad_clip=cfg.train.grad_clip,
        )

        #stop instead of wasting gpu hours if training has broken
        if train_stats["skipped"]:
            print(f"  warning: skipped {train_stats['skipped']} batches with nan/inf loss")
        if train_stats["skipped"] > len(train_loader) // 2:
            raise RuntimeError("More than half the batches had a nan/inf loss; "
                               "stopping. Try a lower train.lr or train.amp=false")

        val_out = validate(model, val_loader, criterion, device, use_amp=use_amp)
        val_metrics = compute_all_metrics(val_out["labels"], val_out["predictions"],
                                          cfg.model.num_classes)

        #one csv row per epoch (plots.py reads this)
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["accuracy"],
            "val_loss": val_out["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_qwk": val_metrics["qwk"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_seconds": round(time.time() - start, 1),
        }
        logger.log(row)
        print(f"epoch {epoch:3d} | train loss {row['train_loss']:.4f} "
              f"acc {row['train_acc']:.3f} | val loss {row['val_loss']:.4f} "
              f"acc {row['val_acc']:.3f} qwk {row['val_qwk']:.4f} | "
              f"{row['epoch_seconds']}s")

        extra = {"epoch": epoch, "val_qwk": val_metrics["qwk"],
                 "config": config_to_dict(cfg)}

        #latest state, to resume
        save_checkpoint(last_path, model, optimizer, scheduler, **extra)

        #best by val QWK (never test)
        if val_metrics["qwk"] > best_qwk:
            best_qwk = val_metrics["qwk"]
            save_checkpoint(best_path, model, **extra)
            print(f"  new best val QWK {best_qwk:.4f} -> saved {best_path}")

    print(f"Done. Best validation QWK: {best_qwk:.4f}")
    return {"best_qwk": best_qwk, "best_path": best_path, "output_dir": out_dir}


#needed on windows for dataloader workers
if __name__ == "__main__":
    main()
