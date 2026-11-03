"""Test the config loads, derived values (J=64, M=65, E=384, cls 32), overrides."""

import os
import sys

#so `import vim` works when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from vim.config import config_from_dict, config_to_dict, load_config

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")


def test_default_config_values():
    cfg = load_config(CONFIG_PATH)

    #scaled down Vim-Ti config
    assert cfg.model.img_size == 128
    assert cfg.model.patch_size == 16
    assert cfg.model.embed_dim == 192
    assert cfg.model.d_state == 16
    assert cfg.model.depth == 12
    assert cfg.model.num_classes == 5
    assert cfg.model.cls_position == "middle"
    assert cfg.model.conv_kernel == 4

    #optimiser values (lr lowered from the paper 1e-3, see yaml)
    assert cfg.train.lr == pytest.approx(5e-4)
    assert cfg.train.weight_decay == pytest.approx(0.05)

    assert isinstance(cfg.seed, int)


def test_derived_values():
    cfg = load_config(CONFIG_PATH)

    print("grid_size  :", cfg.model.grid_size)
    print("num_patches:", cfg.model.num_patches)
    print("seq_len    :", cfg.model.seq_len)
    print("expand_dim :", cfg.model.expand_dim)
    print("cls_index  :", cfg.model.cls_index)

    assert cfg.model.grid_size == 8
    assert cfg.model.num_patches == 64
    assert cfg.model.seq_len == 65
    assert cfg.model.expand_dim == 384
    assert cfg.model.cls_index == 32


def test_overrides_change_value_and_type():
    overrides = ["train.epochs=5", "model.cls_position=head"]
    cfg = load_config(CONFIG_PATH, overrides=overrides)

    #int not "5"
    assert cfg.train.epochs == 5
    assert isinstance(cfg.train.epochs, int)

    #head -> index 0
    assert cfg.model.cls_index == 0


def test_override_typo_is_rejected():
    with pytest.raises(KeyError):
        load_config(CONFIG_PATH, overrides=["train.epochz=5"])


def test_round_trip_through_dict():
    #config -> dict -> config should be equal
    cfg = load_config(CONFIG_PATH)
    again = config_from_dict(config_to_dict(cfg))
    assert again == cfg


if __name__ == "__main__":
    test_default_config_values()
    test_derived_values()
    test_overrides_change_value_and_type()
    test_override_typo_is_rejected()
    test_round_trip_through_dict()
    print("OK")
