"""
selective_scan: the SSM recurrence, Sec 3.1 Eq. 2 and Algorithm 1 lines 13-22.

    h_t = A_bar h_{t-1} + B_bar x_t
    y_t = C h_t
    with A_bar = exp(Delta A) (Eq. 1) and the simplified B_bar = Delta B (line 14).

    Line 13 is printed without exp in the paper, we use exp like Eq. 1 and the official code.
    Plain python loop over the 65 tokens, slow but readable (no CUDA kernel).

"""

import torch


def selective_scan(x, delta, A, B, C):
    """Run Eq. 2 along the sequence.

        x      (Bt, 65, 384)  after conv + SiLU
        delta  (Bt, 65, 384)  step size, > 0
        A      (384, 16)      always negative
        B      (Bt, 65, 16)   write gate
        C      (Bt, 65, 16)   read gate
    returns y  (Bt, 65, 384)
    """
    batch_size, seq_len, channels = x.shape
    #N = 16
    state_dim = A.shape[1]

    assert delta.shape == x.shape, "delta must have the same shape as x: (Bt, M, E)"
    assert A.shape == (channels, state_dim), "A must have shape (E, N)"
    assert B.shape == (batch_size, seq_len, state_dim), "B must have shape (Bt, M, N)"
    assert C.shape == (batch_size, seq_len, state_dim), "C must have shape (Bt, M, N)"

    #(Bt, 65, 384, 1) * (384, 16) -> (Bt, 65, 384, 16)
    delta_A = delta.unsqueeze(-1) * A

    #line 13: A_bar = exp(Delta A), delta > 0 and A < 0 so it stays between 0 and 1
    A_bar = torch.exp(delta_A)

    #line 14: B_bar = Delta B
    #(Bt, 65, 384, 1) * (Bt, 65, 1, 16) -> (Bt, 65, 384, 16)
    B_bar = delta.unsqueeze(-1) * B.unsqueeze(2)

    #line 16: h starts at zero (Bt, 384, 16)
    h = torch.zeros(batch_size, channels, state_dim, device=x.device, dtype=A_bar.dtype)

    #line 17: collect rows in a list instead of filling a zeros tensor (same result)
    ys = []

    #split by token once, unbind is much faster in backward than indexing in the loop
    #A_bar, B_bar -> 65 x (Bt, 384, 16), x -> 65 x (Bt, 384), C -> 65 x (Bt, 16)
    A_bar_steps = torch.unbind(A_bar, dim=1)
    B_bar_steps = torch.unbind(B_bar, dim=1)
    x_steps = torch.unbind(x, dim=1)
    C_steps = torch.unbind(C, dim=1)

    #line 19: for i in 0..M-1, has to be in order since h depends on last step
    for t in range(seq_len):
        #(Bt, 384) -> (Bt, 384, 1)
        x_t = x_steps[t].unsqueeze(-1)

        #line 20: h = A_bar * h + B_bar * x  (forget some, add the new token)
        #A_bar is diagonal so elementwise is enough
        h = A_bar_steps[t] * h + B_bar_steps[t] * x_t

        #(Bt, 16) -> (Bt, 1, 16)
        C_t = C_steps[t].unsqueeze(1)

        #line 21: y = C h, sum over the 16 states -> (Bt, 384)
        y_t = (h * C_t).sum(dim=-1)

        ys.append(y_t)

    #line 22: 65 x (Bt, 384) -> (Bt, 65, 384)
    y = torch.stack(ys, dim=1)

    return y
