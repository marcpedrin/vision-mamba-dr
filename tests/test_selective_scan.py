"""Test selective_scan: shape, the two extremes, a by-hand 2 step case, gradients."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from vim.selective_scan import selective_scan


def make_inputs(batch=2, seq_len=65, channels=384, state_dim=16, seed=0):
    """Random valid inputs: delta > 0, A < 0."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, seq_len, channels, generator=g)
    #0.01 to 0.11
    delta = torch.rand(batch, seq_len, channels, generator=g) * 0.1 + 0.01
    #-1.5 to -0.5
    A = -(torch.rand(channels, state_dim, generator=g) + 0.5)
    B = torch.randn(batch, seq_len, state_dim, generator=g)
    C = torch.randn(batch, seq_len, state_dim, generator=g)
    return x, delta, A, B, C


def test_shape():
    x, delta, A, B, C = make_inputs()

    y = selective_scan(x, delta, A, B, C)

    print("x shape:", tuple(x.shape), " y shape:", tuple(y.shape))

    assert y.shape == (2, 65, 384)


def test_fast_decay_output_depends_only_on_current_token():
    #A = -10000 -> A_bar = 0, everything forgotten each step
    x, _, _, B, C = make_inputs(batch=1, seq_len=10, channels=4, state_dim=3)
    delta = torch.ones(1, 10, 4)
    A = torch.full((4, 3), -10000.0)

    y = selective_scan(x, delta, A, B, C)

    #change only token 0
    x_changed = x.clone()
    x_changed[:, 0] = x_changed[:, 0] + 1000.0
    y_changed = selective_scan(x_changed, delta, A, B, C)

    assert not torch.allclose(y[:, 0], y_changed[:, 0])
    #later tokens dont notice
    assert torch.allclose(y[:, 1:], y_changed[:, 1:])

    #y_t = x_t * sum(B * C)
    expected = x * (B * C).sum(dim=-1).unsqueeze(-1)
    assert torch.allclose(y, expected, atol=1e-5)


def test_no_decay_accumulates_like_running_sum():
    #A ~ 0 -> A_bar ~ 1, nothing leaks, with B = C = 1 y is the running sum
    x = torch.randn(2, 20, 5)
    delta = torch.ones(2, 20, 5)
    A = torch.full((5, 1), -1e-8)
    B = torch.ones(2, 20, 1)
    C = torch.ones(2, 20, 1)

    y = selective_scan(x, delta, A, B, C)

    expected = torch.cumsum(x, dim=1)
    assert torch.allclose(y, expected, atol=1e-4)


def test_hand_computed_two_steps():
    #1 image, 2 tokens, 1 channel, 2 states
    #token 0: B_bar = 0.5*[1, 0] = [0.5, 0], h = [0.5, 0], y = 1*0.5 + 1*0 = 0.5
    #token 1: A_bar = [e^-1, e^-2], B_bar = [0.5, 1]
    #         h = [0.5e^-1 + 1, 2], y = 2(0.5e^-1 + 1) - 2 = e^-1
    x = torch.tensor([[[1.0], [2.0]]])
    delta = torch.tensor([[[0.5], [1.0]]])
    A = torch.tensor([[-1.0, -2.0]])
    B = torch.tensor([[[1.0, 0.0], [0.5, 1.0]]])
    C = torch.tensor([[[1.0, 1.0], [2.0, -1.0]]])

    y = selective_scan(x, delta, A, B, C)

    expected = torch.tensor([[[0.5], [math.exp(-1.0)]]])
    print("y =", y.flatten().tolist(), " expected =", expected.flatten().tolist())
    assert torch.allclose(y, expected, atol=1e-6)


def test_gradients_reach_every_input():
    x, delta, A, B, C = make_inputs(batch=2, seq_len=8, channels=6, state_dim=4)

    for tensor in (x, delta, A, B, C):
        tensor.requires_grad_()

    y = selective_scan(x, delta, A, B, C)
    y.sum().backward()

    #every input gets a finite non zero gradient
    for name, tensor in [("x", x), ("delta", delta), ("A", A), ("B", B), ("C", C)]:
        assert tensor.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(tensor.grad).all(), f"{name} has non-finite gradient"
        assert tensor.grad.abs().sum() > 0, f"{name} has an all-zero gradient"


if __name__ == "__main__":
    test_shape()
    test_fast_decay_output_depends_only_on_current_token()
    test_no_decay_accumulates_like_running_sum()
    test_hand_computed_two_steps()
    test_gradients_reach_every_input()
    print("OK")
