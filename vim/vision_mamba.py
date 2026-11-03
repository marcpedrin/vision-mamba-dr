"""
VisionMamba: the full classifier, Sec 3.2 and 3.4.

    image -> patches -> tokens (Eq. 4) -> 12 Vim blocks -> norm -> class token -> head
    (B, 3, 128, 128) -> (B, 64, 192) -> (B, 65, 192) -> (B, 192) -> (B, 5)

    12 blocks instead of 24 and 128px instead of 224 to fit one GPU.
    Head is a single Linear like the official code.

"""

import torch.nn as nn

from .patch_embed import PatchEmbed
from .tokens import ClassTokenPosEmbed
from .vim_block import VimBlock


class VisionMamba(nn.Module):
    #(B, 3, 128, 128) -> (B, 5) logits

    def __init__(self, img_size=128, patch_size=16, in_chans=3, embed_dim=192,
                 expand=2, d_state=16, depth=12, num_classes=5, cls_index=32,
                 conv_kernel=4, dt_min=0.001, dt_max=0.1, d_skip=False):
        super().__init__()

        #image -> (B, 64, 192)
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.cls_index = cls_index

        #class token + E_pos -> (B, 65, 192)
        self.tokens = ClassTokenPosEmbed(
            num_patches=self.patch_embed.num_patches,
            embed_dim=embed_dim,
            cls_index=cls_index,
        )

        #12 blocks, each with its own weights
        #ModuleList so pytorch can see the weights (plain list wouldnt be trained)
        blocks = []
        for _ in range(depth):
            blocks.append(
                VimBlock(
                    embed_dim=embed_dim,
                    expand=expand,
                    d_state=d_state,
                    conv_kernel=conv_kernel,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    d_skip=d_skip,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        #f = Norm(T_L^0)
        self.norm = nn.LayerNorm(embed_dim)

        #192 -> 5 grade scores
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, images):
        #(B, 3, 128, 128) -> (B, 64, 192)
        x = self.patch_embed(images)

        #(B, 64, 192) -> (B, 65, 192)
        x = self.tokens(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        #only keep the class token: (B, 65, 192) -> (B, 192)
        cls_output = x[:, self.cls_index]

        #(B, 192) -> (B, 5)
        logits = self.head(cls_output)

        return logits


def build_model(model_cfg):
    """Build VisionMamba from the model section of the config."""
    return VisionMamba(
        img_size=model_cfg.img_size,
        patch_size=model_cfg.patch_size,
        in_chans=model_cfg.in_chans,
        embed_dim=model_cfg.embed_dim,
        expand=model_cfg.expand,
        d_state=model_cfg.d_state,
        depth=model_cfg.depth,
        num_classes=model_cfg.num_classes,
        cls_index=model_cfg.cls_index,
        conv_kernel=model_cfg.conv_kernel,
        dt_min=model_cfg.dt_min,
        dt_max=model_cfg.dt_max,
        d_skip=model_cfg.d_skip,
    )


def count_parameters(module):
    """Count trainable parameters. Default model: 6,857,093.

    Not half of Vim-Ti's 7M even with half the blocks, because line 11's Delta
    projection is a full 384x384 (official code uses rank 12).
    """
    total = 0
    for p in module.parameters():
        if p.requires_grad:
            total += p.numel()
    return total


#python -m vim.vision_mamba prints the breakdown
if __name__ == "__main__":
    model = VisionMamba()
    print(f"Total parameters: {count_parameters(model):,}")
    for name, child in model.named_children():
        print(f"  {name:12s} {count_parameters(child):>10,}")
    block = model.blocks[0]
    print("One VimBlock:")
    for name, child in block.named_children():
        print(f"  {name:12s} {count_parameters(child):>10,}")
