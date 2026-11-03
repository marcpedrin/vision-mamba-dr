"""Test ClassTokenPosEmbed: (2, 64, 192) -> (2, 65, 192), class token at 32, params registered."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from vim.tokens import ClassTokenPosEmbed


def test_tokens_shape():
    module = ClassTokenPosEmbed()

    #fake PatchEmbed output
    x = torch.randn(2, 64, 192)

    out = module(x)

    print("Input shape :", tuple(x.shape))
    print("Output shape:", tuple(out.shape))

    #64 patches + 1 class token
    assert out.shape == (2, 65, 192)


def test_class_token_lands_at_index_32():
    module = ClassTokenPosEmbed()

    #pos embed 0 and class token 100 so its easy to spot
    with torch.no_grad():
        module.pos_embed.zero_()
        module.cls_token.fill_(100.0)

    #patch j is all j's
    x = torch.arange(64).float().view(1, 64, 1).expand(2, 64, 192)

    out = module(x)

    assert torch.all(out[:, 32] == 100.0)
    #before it unchanged
    assert torch.all(out[:, 31] == 31.0)
    assert torch.all(out[:, 0] == 0.0)
    #after it shifted by one
    assert torch.all(out[:, 33] == 32.0)
    assert torch.all(out[:, 64] == 63.0)


def test_head_position_puts_class_token_first():
    #ViT style head, Table 5
    module = ClassTokenPosEmbed(cls_index=0)
    with torch.no_grad():
        module.pos_embed.zero_()
        module.cls_token.fill_(100.0)

    out = module(torch.zeros(1, 64, 192))

    assert torch.all(out[:, 0] == 100.0)


def test_parameters_are_registered():
    module = ClassTokenPosEmbed()

    shapes = {name: tuple(p.shape) for name, p in module.named_parameters()}
    print("Parameters:", shapes)

    assert shapes["cls_token"] == (1, 1, 192)
    assert shapes["pos_embed"] == (1, 65, 192)

    for p in module.parameters():
        assert p.requires_grad

    #small random init within +-0.04
    assert module.pos_embed.abs().max() > 0
    assert module.pos_embed.abs().max() <= 0.04 + 1e-6


if __name__ == "__main__":
    test_tokens_shape()
    test_class_token_lands_at_index_32()
    test_head_position_puts_class_token_first()
    test_parameters_are_registered()
    print("OK")
