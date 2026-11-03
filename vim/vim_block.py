"""
VimBlock: one bidirectional Vim block, Sec 3.3 and all of Algorithm 1.

    norm -> split into x and z -> scan forward and backward (own weights each)
    -> gate both with SiLU(z) -> add -> project back -> + residual
    (B, 65, 192) -> (B, 65, 192), E = 384 and N = 16 inside

    Differences from official code: full 384x384 Delta projection like line 11
    (official is low rank), LayerNorm instead of RMSNorm, no /2.
    The official D*x skip is included and on by default (model.d_skip).

"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .selective_scan import selective_scan


class DirectionalSSM(nn.Module):
    #Algorithm 1 lines 7-22 for one direction. (B, 65, 384) -> (B, 65, 384)

    def __init__(self, expand_dim=384, d_state=16, conv_kernel=4, dt_min=0.001,
                 dt_max=0.1, d_skip=True):
        super().__init__()

        self.expand_dim = expand_dim

        #line 7: depthwise conv (groups = channels so each channel only sees itself)
        #padding 3 then crop the end in forward -> causal, token t sees t-3..t
        self.conv1d = nn.Conv1d(
            in_channels=expand_dim,
            out_channels=expand_dim,
            kernel_size=conv_kernel,
            groups=expand_dim,
            padding=conv_kernel - 1,
        )

        #lines 8-9: B and C from each token, 384 -> 16
        self.proj_B = nn.Linear(expand_dim, d_state, bias=False)
        self.proj_C = nn.Linear(expand_dim, d_state, bias=False)

        #line 11: Delta projection 384 -> 384 + its own bias Parameter^Delta
        self.proj_delta = nn.Linear(expand_dim, expand_dim, bias=False)
        self.delta_bias = nn.Parameter(torch.zeros(expand_dim))

        #line 12: store log(-A) so A = -exp(A_log) is always negative
        self.A_log = nn.Parameter(torch.zeros(expand_dim, d_state))

        #official Mamba skip y + D*x (not in printed Algorithm 1)
        #without it the class token barely sees the image at the start and training stalls
        self.D = nn.Parameter(torch.ones(expand_dim)) if d_skip else None

        self._init_weights(d_state, dt_min, dt_max)

    def _init_weights(self, d_state, dt_min, dt_max):
        with torch.no_grad():
            #A = -1, -2, ..., -16 for each channel (S4D-real init)
            rates = torch.arange(1, d_state + 1, dtype=torch.float32)
            self.A_log.copy_(torch.log(rates).unsqueeze(0).expand_as(self.A_log))

            #random step size per channel between 0.001 and 0.1 on a log scale
            log_min = math.log(dt_min)
            log_max = math.log(dt_max)
            uniform = torch.rand(self.expand_dim)
            dt = torch.exp(uniform * (log_max - log_min) + log_min)
            #inverse softplus so softplus(delta_bias) = dt at the start
            self.delta_bias.copy_(torch.log(torch.expm1(dt)))

    def forward(self, x):
        #x (B, 65, 384), already flipped for backward
        assert x.shape[2] == self.expand_dim, (
            f"Expected {self.expand_dim} channels, got {x.shape[2]}"
        )
        seq_len = x.shape[1]

        #conv1d wants channels first: (B, 65, 384) -> (B, 384, 65)
        x_conv = x.transpose(1, 2)

        #(B, 384, 65) -> (B, 384, 68) because of padding
        x_conv = self.conv1d(x_conv)

        #keep first 65 so its causal
        x_conv = x_conv[:, :, :seq_len]

        #back to (B, 65, 384)
        x_conv = x_conv.transpose(1, 2)

        #line 7: SiLU
        x_conv = F.silu(x_conv)

        #lines 8-9: (B, 65, 384) -> (B, 65, 16)
        B = self.proj_B(x_conv)
        C = self.proj_C(x_conv)

        #line 11: Delta = softplus(Linear(x') + bias), always positive
        delta_raw = self.proj_delta(x_conv) + self.delta_bias
        delta = F.softplus(delta_raw)

        #line 12: A (384, 16), negative
        A = -torch.exp(self.A_log)

        #lines 13-22
        y = selective_scan(x_conv, delta, A, B, C)

        if self.D is not None:
            y = y + self.D * x_conv

        return y


class VimBlock(nn.Module):
    #Algorithm 1 lines 1-28. (B, 65, 192) -> (B, 65, 192)

    def __init__(self, embed_dim=192, expand=2, d_state=16, conv_kernel=4,
                 dt_min=0.001, dt_max=0.1, d_skip=True):
        super().__init__()

        #E = 2 * 192 = 384
        expand_dim = expand * embed_dim
        self.embed_dim = embed_dim
        self.expand_dim = expand_dim

        #line 2: Norm
        self.norm = nn.LayerNorm(embed_dim)

        #lines 3-4: Linear^x and Linear^z as one 192 -> 768, split in forward
        self.in_proj = nn.Linear(embed_dim, 2 * expand_dim, bias=False)

        #line 6: separate weights for each direction
        self.forward_ssm = DirectionalSSM(expand_dim, d_state, conv_kernel, dt_min,
                                          dt_max, d_skip)
        self.backward_ssm = DirectionalSSM(expand_dim, d_state, conv_kernel, dt_min,
                                           dt_max, d_skip)

        #line 28: Linear^T, 384 -> 192
        self.out_proj = nn.Linear(expand_dim, embed_dim, bias=False)

    def forward(self, tokens):
        #T_{l-1} (B, 65, 192), saved for the residual
        residual = tokens

        #line 2
        normed = self.norm(tokens)

        #lines 3-4: (B, 65, 192) -> (B, 65, 768)
        xz = self.in_proj(normed)

        #x = first 384, z = last 384
        x = xz[..., : self.expand_dim]
        z = xz[..., self.expand_dim :]

        y = {}

        #line 6: for o in {forward, backward}
        for direction in ("forward", "backward"):
            if direction == "forward":
                #reads tokens 0 -> 64
                y[direction] = self.forward_ssm(x)
            else:
                #flip, run the same code with backward weights, flip back
                x_reversed = torch.flip(x, dims=[1])
                y_reversed = self.backward_ssm(x_reversed)
                y[direction] = torch.flip(y_reversed, dims=[1])

        #lines 25-26: gate with SiLU(z)
        gate = F.silu(z)
        y_forward_gated = y["forward"] * gate
        y_backward_gated = y["backward"] * gate

        #line 28: add both directions, (B, 65, 384) -> (B, 65, 192)
        out = self.out_proj(y_forward_gated + y_backward_gated)

        #line 28: + T_{l-1} residual
        return out + residual
