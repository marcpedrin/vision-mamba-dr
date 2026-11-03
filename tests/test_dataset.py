"""Test the data pipeline on fake JPEGs in a temp folder (tmp_path is a pytest fixture)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from PIL import Image

from data import build_dataloaders, build_messidor_loader
from data.eyepacs import EyePACSDataset, read_eyepacs_labels
from data.splits import split_by_patient
from data.synthetic import make_synthetic_eyepacs, make_synthetic_messidor
from data.transforms import CropToRetina, build_eval_transform
from vim.config import Config


def make_config(image_dir, labels_csv):
    """Default config pointed at the fake data."""
    cfg = Config()
    cfg.data.eyepacs_image_dir = image_dir
    cfg.data.eyepacs_labels_csv = labels_csv
    cfg.data.num_workers = 0
    return cfg


def test_dataset_length_and_item(tmp_path):
    #10 patients x 2 eyes
    image_dir, labels_csv = make_synthetic_eyepacs(str(tmp_path), num_patients=10)
    labels = read_eyepacs_labels(labels_csv)

    dataset = EyePACSDataset(image_dir, labels, build_eval_transform(img_size=128))

    assert len(dataset) == 20

    image, label = dataset[0]
    print("Item image shape:", tuple(image.shape), " label:", label)
    assert image.shape == (3, 128, 128)
    assert isinstance(label, int)
    assert 0 <= label <= 4
    #"1_left" -> "1"
    assert labels["patient_id"].iloc[0] == "1"


def test_missing_image_is_skipped_with_warning(tmp_path):
    image_dir, labels_csv = make_synthetic_eyepacs(str(tmp_path), num_patients=5)
    labels = read_eyepacs_labels(labels_csv)

    #delete one that the csv still lists
    os.remove(os.path.join(image_dir, "3_left.jpeg"))

    with pytest.warns(UserWarning):
        dataset = EyePACSDataset(image_dir, labels, build_eval_transform())

    #10 - 1 = 9
    assert len(dataset) == 9
    for i in range(len(dataset)):
        image, _ = dataset[i]
        assert image.shape == (3, 128, 128)


def test_crop_removes_black_border():
    #white 100x80 box on black
    image = Image.new("RGB", (300, 200), (0, 0, 0))
    image.paste((255, 255, 255), (120, 60, 220, 140))

    cropped = CropToRetina(threshold=10)(image)

    assert cropped.size == (100, 80)


def test_splits_are_disjoint_and_patient_grouped(tmp_path):
    _, labels_csv = make_synthetic_eyepacs(str(tmp_path), num_patients=50)
    labels = read_eyepacs_labels(labels_csv)

    train_df, val_df, test_df = split_by_patient(labels, 0.1, 0.1, seed=42)
    print("Split sizes (images):", len(train_df), len(val_df), len(test_df))

    train_imgs = set(train_df["image"])
    val_imgs = set(val_df["image"])
    test_imgs = set(test_df["image"])

    train_pat = set(train_df["patient_id"])
    val_pat = set(val_df["patient_id"])
    test_pat = set(test_df["patient_id"])

    #no image in two splits
    assert not (train_imgs & val_imgs)
    assert not (train_imgs & test_imgs)
    assert not (val_imgs & test_imgs)

    #no patient in two splits
    assert not (train_pat & val_pat)
    assert not (train_pat & test_pat)
    assert not (val_pat & test_pat)

    #covers everything exactly once
    assert len(train_df) + len(val_df) + len(test_df) == len(labels)
    assert train_imgs | val_imgs | test_imgs == set(labels["image"])

    assert len(val_df) > 0 and len(test_df) > 0
    assert len(train_df) > len(val_df)


def test_split_is_reproducible(tmp_path):
    _, labels_csv = make_synthetic_eyepacs(str(tmp_path), num_patients=50)
    labels = read_eyepacs_labels(labels_csv)

    #same seed same split, different seed different split
    first = split_by_patient(labels, 0.1, 0.1, seed=42)
    second = split_by_patient(labels, 0.1, 0.1, seed=42)
    other = split_by_patient(labels, 0.1, 0.1, seed=7)

    assert list(first[1]["image"]) == list(second[1]["image"])
    assert list(first[1]["image"]) != list(other[1]["image"])


def test_dataloader_batch_shape(tmp_path):
    image_dir, labels_csv = make_synthetic_eyepacs(str(tmp_path), num_patients=20)
    cfg = make_config(image_dir, labels_csv)

    train_loader, val_loader, test_loader = build_dataloaders(cfg, batch_size=4)

    #first batch
    images, labels = next(iter(train_loader))
    print("Batch shapes:", tuple(images.shape), tuple(labels.shape))

    assert images.shape == (4, 3, 128, 128)
    assert labels.shape == (4,)
    assert labels.dtype == torch.int64


def test_messidor_skips_ungradable(tmp_path):
    #6 gradable + 1 ungradable
    image_dir, labels_csv = make_synthetic_messidor(str(tmp_path), num_images=6)
    cfg = Config()
    cfg.data.messidor_image_dir = image_dir
    cfg.data.messidor_labels_csv = labels_csv
    cfg.data.num_workers = 0

    loader = build_messidor_loader(cfg, batch_size=3)

    assert len(loader.dataset) == 6
    images, labels = next(iter(loader))
    assert images.shape == (3, 3, 128, 128)


if __name__ == "__main__":
    import tempfile

    test_crop_removes_black_border()
    folder_tests = [
        test_dataset_length_and_item,
        test_missing_image_is_skipped_with_warning,
        test_splits_are_disjoint_and_patient_grouped,
        test_split_is_reproducible,
        test_dataloader_batch_shape,
        test_messidor_skips_ungradable,
    ]
    for test in folder_tests:
        with tempfile.TemporaryDirectory() as folder:
            test(folder)
    print("OK")
