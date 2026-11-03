"""
Training and validation loops, one epoch each.

    train_one_epoch: goes over every training image once and updates the weights
    validate: runs the model on the val set with no updates, returns predictions

"""

import numpy as np
import torch
from tqdm import tqdm

from .utils import AverageMeter


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device,
                    scaler, use_amp=False, grad_clip=None):
    """One pass over the train loader. Returns {"loss", "accuracy"}."""
    #training mode (no dropout/batchnorm here but good habit)
    model.train()

    loss_meter = AverageMeter()
    correct = 0
    seen = 0
    skipped = 0

    for images, labels in tqdm(loader, desc="train", leave=False):
        #images (B, 3, 128, 128), labels (B,)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        #1. clear old gradients (pytorch adds them up otherwise)
        optimizer.zero_grad(set_to_none=True)

        #mixed precision on gpu, does nothing on cpu
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=use_amp):
            #2. forward (B, 3, 128, 128) -> (B, 5)
            logits = model(images)
            #3. weighted cross entropy -> one number
            loss = criterion(logits, labels)

        #nan/inf loss -> skip the batch so it cant break the weights
        if not torch.isfinite(loss):
            skipped += 1
            scheduler.step()
            continue

        #4. backward, scaler stops tiny fp16 gradients going to zero
        scaler.scale(loss).backward()

        #clip the gradient so one bad batch cant blow the weights up
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        #5. update weights (AdamW)
        scaler.step(optimizer)
        scaler.update()

        #lr schedule moves every batch
        scheduler.step()

        batch_size = labels.shape[0]
        loss_meter.update(loss.item(), n=batch_size)
        #predicted grade = highest of the 5 scores
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        seen += batch_size

    #max(seen, 1) so empty loader doesnt divide by zero
    return {"loss": loss_meter.avg, "accuracy": correct / max(seen, 1),
            "skipped": skipped}


#no_grad: no graph recorded, saves memory since we never call backward here
@torch.no_grad()
def validate(model, loader, criterion, device, use_amp=False):
    """Evaluate on loader. Returns mean loss + true and predicted grades."""
    model.eval()

    loss_meter = AverageMeter()
    all_labels = []
    all_predictions = []

    for images, labels in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        #forward only: (B, 3, 128, 128) -> (B, 5)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        loss_meter.update(loss.item(), n=labels.shape[0])

        #(B, 5) -> (B,)
        predictions = logits.argmax(dim=1)

        #back to cpu as numpy
        all_labels.append(labels.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())

    #join batches into one array each
    return {
        "loss": loss_meter.avg,
        "labels": np.concatenate(all_labels),
        "predictions": np.concatenate(all_predictions),
    }
