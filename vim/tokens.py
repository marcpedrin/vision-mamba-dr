"""
ClassTokenPosEmbed: adds the class token and position embeddings.

    Eq. 4: T_0 = [t_p^1 W ; ... ; t_cls ; ... ; t_p^J W] + E_pos
    Class token goes in the MIDDLE (index 32), paper Table 5: 76.1% vs 75.2% for head.
    (B, 64, 192) -> (B, 65, 192)

"""

import torch
import torch.nn as nn


class ClassTokenPosEmbed(nn.Module):
    #Insert a learnable class token at cls_index, then add learnable E_pos.

    def __init__(self, num_patches=64, embed_dim=192, cls_index=32):
        super().__init__()

        assert 0 <= cls_index <= num_patches, "cls_index must be in 0..num_patches"

        self.num_patches = num_patches
        self.cls_index = cls_index

        #t_cls: (1, 1, 192), shared by every image in the batch
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        #E_pos: one vector per position, 64 + 1 = 65 -> (1, 65, 192)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self._init_weights()

    def _init_weights(self):
        #small random values, std 0.02, cut at +-2 std
        #a and b are absolute values not multiples of std
        std = 0.02
        nn.init.trunc_normal_(self.cls_token, std=std, a=-2 * std, b=2 * std)
        nn.init.trunc_normal_(self.pos_embed, std=std, a=-2 * std, b=2 * std)

    def forward(self, patch_tokens):
        #patch_tokens (B, 64, 192) from PatchEmbed
        assert patch_tokens.shape[1] == self.num_patches, (
            f"Expected {self.num_patches} patch tokens, got {patch_tokens.shape[1]}"
        )

        batch_size = patch_tokens.shape[0]

        #(1, 1, 192) -> (B, 1, 192), expand doesnt copy memory
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        #split 64 patches into first 32 and last 32
        before = patch_tokens[:, : self.cls_index]
        after = patch_tokens[:, self.cls_index :]

        #(B, 32, 192) + (B, 1, 192) + (B, 32, 192) -> (B, 65, 192)
        tokens = torch.cat([before, cls_tokens, after], dim=1)

        #broadcast (1, 65, 192) over the batch
        tokens = tokens + self.pos_embed

        #T_0, goes into the first block
        return tokens
