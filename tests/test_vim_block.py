"""Test VimBlock: shape, separate weights per direction, direction aware, 12 block stack, gradients."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F

from vim.vim_block import VimBlock


def test_vim_block_shape():
    torch.manual_seed(0)
    block = VimBlock()

    #64 patches + class token
    x = torch.randn(2, 65, 192)
    out = block(x)

    print("Input shape :", tuple(x.shape))
    print("Output shape:", tuple(out.shape))

    #same shape so blocks can stack
    assert out.shape == (2, 65, 192)


def test_directions_have_separate_parameters():
    torch.manual_seed(0)
    block = VimBlock()

    forward_params = dict(block.forward_ssm.named_parameters())
    backward_params = dict(block.backward_ssm.named_parameters())

    #same layers
    assert forward_params.keys() == backward_params.keys()
    print("Per-direction parameters:", sorted(forward_params.keys()))

    #but different tensors in memory
    for name in forward_params:
        assert forward_params[name] is not backward_params[name]
        assert forward_params[name].data_ptr() != backward_params[name].data_ptr()

    #and different values
    assert not torch.equal(forward_params["conv1d.weight"],
                           backward_params["conv1d.weight"])
    assert not torch.equal(forward_params["proj_delta.weight"],
                           backward_params["proj_delta.weight"])


def test_block_is_direction_aware():
    torch.manual_seed(0)
    block = VimBlock()
    block.eval()

    x = torch.randn(2, 65, 192)
    #flip token order
    def reverse(t):
        return torch.flip(t, dims=[1])

    with torch.no_grad():
        out_of_reversed = block(reverse(x))
        reversed_out = reverse(block(x))

    #different weights per direction -> not just a mirror
    assert not torch.allclose(out_of_reversed, reversed_out, atol=1e-4)

    #copy forward weights into backward -> now it is a mirror
    block.backward_ssm.load_state_dict(
        block.forward_ssm.state_dict()
    )
    with torch.no_grad():
        out_of_reversed = block(reverse(x))
        reversed_out = reverse(block(x))
    assert torch.allclose(out_of_reversed, reversed_out, atol=1e-5)


def test_stack_of_12_blocks_stays_in_a_sane_range():
    torch.manual_seed(0)
    stack = nn.Sequential(*[VimBlock() for _ in range(12)])
    stack.eval()

    x = torch.randn(2, 65, 192)

    #fresh init: small Delta so the stack is close to identity
    with torch.no_grad():
        out = stack(x)
    std = out.std().item()
    added = (out - x).std().item()
    print(f"Fresh init : output std {std:.3f}, part added by 12 blocks {added:.5f}")
    assert torch.isfinite(out).all()
    assert 0.5 < std < 10.0

    #Delta pushed to ~1 so the blocks actually do something
    awake_bias = torch.log(torch.expm1(torch.tensor(1.0)))
    with torch.no_grad():
        for block in stack:
            block.forward_ssm.delta_bias.fill_(awake_bias)
            block.backward_ssm.delta_bias.fill_(awake_bias)
        out = stack(x)
    std = out.std().item()
    added = (out - x).std().item()
    print(f"Awake Delta: output std {std:.3f}, part added by 12 blocks {added:.5f}")
    assert torch.isfinite(out).all()
    assert 0.5 < std < 10.0
    assert added > 0.01


def test_gradient_flow_through_stack():
    torch.manual_seed(0)
    stack = nn.Sequential(*[VimBlock() for _ in range(12)])

    #smaller sequence to keep it quick
    x = torch.randn(1, 17, 192, requires_grad=True)
    out = stack(x)
    out.sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()

    #every parameter in every block gets a gradient
    for name, p in stack.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"
        assert p.grad.abs().sum() > 0, f"{name} has an all-zero gradient"


def test_A_is_negative_and_delta_starts_in_range():
    torch.manual_seed(0)
    ssm = VimBlock().forward_ssm

    #A negative, -1..-16 at init
    A = -torch.exp(ssm.A_log)
    assert (A < 0).all()
    assert torch.allclose(A[0], -torch.arange(1, 17).float())

    #starting Delta in [0.001, 0.1]
    delta0 = F.softplus(ssm.delta_bias)
    assert (delta0 >= 0.001 - 1e-6).all() and (delta0 <= 0.1 + 1e-6).all()


if __name__ == "__main__":
    test_vim_block_shape()
    test_directions_have_separate_parameters()
    test_block_is_direction_aware()
    test_stack_of_12_blocks_stays_in_a_sane_range()
    test_gradient_flow_through_stack()
    test_A_is_negative_and_delta_starts_in_range()
    print("OK")
