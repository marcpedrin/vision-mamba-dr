"""
Data package: builds the DataLoaders.

    build_dataloaders(cfg) -> train, val, test loaders for EyePACS
    build_messidor_loader(cfg) -> Messidor-2 loader (eval only)
    each batch: images (B, 3, 128, 128), labels (B,)

"""

import torch
from torch.utils.data import DataLoader

from train.utils import seed_worker

from .eyepacs import EyePACSDataset, read_eyepacs_labels
from .messidor2 import Messidor2Dataset
from .splits import split_by_patient
from .transforms import build_eval_transform, build_train_transform


def _make_loader(dataset, batch_size, shuffle, data_cfg, seed):
    """DataLoader with our reproducible settings."""
    #pin memory only helps with a gpu
    pin_memory = data_cfg.pin_memory and torch.cuda.is_available()

    #seeded so the shuffle order is the same every run
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=data_cfg.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_dataloaders(cfg, max_images=None, batch_size=None):
    """EyePACS train/val/test loaders. max_images used by --smoke-test."""
    #columns: image, level, patient_id
    labels = read_eyepacs_labels(cfg.data.eyepacs_labels_csv)

    if max_images is not None:
        labels = labels.head(max_images)

    #split by patient so both eyes stay in the same set
    train_df, val_df, test_df = split_by_patient(
        labels,
        val_fraction=cfg.data.val_fraction,
        test_fraction=cfg.data.test_fraction,
        seed=cfg.seed,
    )

    #augmentation only for train
    train_tf = build_train_transform(
        img_size=cfg.model.img_size,
        crop_threshold=cfg.data.crop_threshold,
        rotation_degrees=cfg.data.rotation_degrees,
        color_jitter=cfg.data.color_jitter,
    )
    eval_tf = build_eval_transform(
        img_size=cfg.model.img_size,
        crop_threshold=cfg.data.crop_threshold,
    )

    image_dir = cfg.data.eyepacs_image_dir
    ext = cfg.data.eyepacs_image_ext
    train_set = EyePACSDataset(image_dir, train_df, train_tf, image_ext=ext)
    val_set = EyePACSDataset(image_dir, val_df, eval_tf, image_ext=ext)
    test_set = EyePACSDataset(image_dir, test_df, eval_tf, image_ext=ext)

    bs = batch_size if batch_size is not None else cfg.train.batch_size

    #only shuffle train
    train_loader = _make_loader(train_set, bs, True, cfg.data, cfg.seed)
    val_loader = _make_loader(val_set, bs, False, cfg.data, cfg.seed)
    test_loader = _make_loader(test_set, bs, False, cfg.data, cfg.seed)

    return train_loader, val_loader, test_loader


def build_messidor_loader(cfg, batch_size=None):
    """Messidor-2 loader, eval only."""
    eval_tf = build_eval_transform(
        img_size=cfg.model.img_size,
        crop_threshold=cfg.data.crop_threshold,
    )
    dataset = Messidor2Dataset(
        cfg.data.messidor_image_dir,
        cfg.data.messidor_labels_csv,
        eval_tf,
        id_column=cfg.data.messidor_id_column,
        label_column=cfg.data.messidor_label_column,
        default_ext=cfg.data.messidor_image_ext,
    )
    bs = batch_size if batch_size is not None else cfg.train.batch_size
    return _make_loader(dataset, bs, False, cfg.data, cfg.seed)
