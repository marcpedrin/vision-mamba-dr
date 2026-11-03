"""Test VisionMamba: (2, 3, 128, 128) -> (2, 5), param count, one step lowers loss, eval mode."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from vim import VisionMamba, build_model, count_parameters
from vim.config import load_config

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")


def test_vision_mamba_shape():
    torch.manual_seed(0)
    model = VisionMamba()

    images = torch.randn(2, 3, 128, 128)
    logits = model(images)

    print("Input shape :", tuple(images.shape))
    print("Output shape:", tuple(logits.shape))

    #5 grades per image
    assert logits.shape == (2, 5)


def test_parameter_count():
    model = VisionMamba()
    total = count_parameters(model)
    print(f"Parameters: {total:,}  (paper Vim-Ti: about 7M with 24 blocks)")

    #if this changes the architecture changed
    assert total == 6_866_309


def test_one_optimiser_step_reduces_loss():
    torch.manual_seed(0)
    #2 blocks to keep it fast
    model = VisionMamba(depth=2)
    model.train()

    images = torch.randn(4, 3, 128, 128)
    labels = torch.tensor([0, 1, 2, 3])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

    loss_before = F.cross_entropy(model(images), labels)

    optimizer.zero_grad()
    loss_before.backward()
    optimizer.step()

    #same batch after one step
    with torch.no_grad():
        loss_after = F.cross_entropy(model(images), labels)

    print(f"Loss before: {loss_before.item():.4f}  after: {loss_after.item():.4f}")
    assert loss_after.item() < loss_before.item()


def test_eval_mode_under_no_grad():
    torch.manual_seed(0)
    model = VisionMamba(depth=2)
    model.eval()

    images = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out1 = model(images)
        out2 = model(images)

    #no graph, and same input same output
    assert not out1.requires_grad
    assert torch.equal(out1, out2)
    assert out1.shape == (2, 5)


def test_build_model_from_config():
    cfg = load_config(CONFIG_PATH)
    model = build_model(cfg.model)

    assert len(model.blocks) == cfg.model.depth
    assert model.cls_index == 32
    assert count_parameters(model) == count_parameters(VisionMamba(d_skip=cfg.model.d_skip))


if __name__ == "__main__":
    test_vision_mamba_shape()
    test_parameter_count()
    test_one_optimiser_step_reduces_loss()
    test_eval_mode_under_no_grad()
    test_build_model_from_config()
    print("OK")
