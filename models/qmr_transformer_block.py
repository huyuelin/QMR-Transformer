"""
QMR-Transformer Block (paper §3.3–3.4, Eq. 1–3).

Core building block of QMR-Transformers.  Each block contains:
  - Routing heads: sparse attention restricted to mixed-radix offsets,
    logits biased by compiler prior (Eq. 1)
  - Local content heads: dense attention within a sliding window
  - Scalar gates g^r, g^c controlling the residual mix
  - Optional FFN update
  - Optional subspace splitting (Full++) and perturbation penalty

Routing-head logit (Eq. 1):
    s_{l,h}(i,j) = (W^Q h_i)^T (W^K h_j) / sqrt(d_h)
                 + ρ_{l,h}(j - i)                        [relative bias]
                 + λ_{l,h} * log(π_{l,h}(a(i,j)|q) + ε) [compiler prior]

QMR-Core+ update (§3.4):
    h_i^{l+1} = h_i^l + g_l^r * o_route^l(i)
                       + g_l^c * o_local^l(i)
                       + FFN_l(h_i^l)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from models.mixed_radix_generator import MixedRadixGraphGenerator


class QMRRoutingHead(nn.Module):
    """
    Implements the routing-head logit formula (paper Eq. 1).

    Candidate set at position i (layer l):
        N_l(i) = {i - a * B_{l-1} : a = 0,...,b_l-1} ∩ [0, L)

    Logit:
        s_{l,h}(i,j) = QK^T / sqrt(d_h) + rho(j-i) + lambda * log(pi(a|q) + eps)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        layer_idx: int,
        max_b: int = 64,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_h = d_model // num_heads
        self.layer_idx = layer_idx
        self.eps = eps

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # Relative position bias: rho_{l,h}(a), one per (head, digit a)
        self.rho = nn.Parameter(torch.zeros(num_heads, max_b))

        # Compiler coupling strength: lambda_{l,h} (per head)
        self.lam = nn.Parameter(torch.ones(num_heads))

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,              # (batch, L, d_model)
        compiler_dists: torch.Tensor, # (batch, b_l)
        offsets: torch.Tensor,        # (b_l,) signed offsets
        B_prev: int,
        causal: bool = False,
    ) -> torch.Tensor:
        """
        Returns:
            output: (batch, L, d_model)
        """
        batch, L, _ = h.shape
        b = offsets.shape[0]
        H = self.num_heads
        d_h = self.d_h
        device = h.device

        Q = self.W_Q(h).view(batch, L, H, d_h).permute(0, 2, 1, 3)  # (B,H,L,d_h)
        K = self.W_K(h).view(batch, L, H, d_h).permute(0, 2, 1, 3)
        V = self.W_V(h).view(batch, L, H, d_h).permute(0, 2, 1, 3)

        # Build candidate position matrix j[i, a] = i + offsets[a]
        i_range = torch.arange(L, device=device)
        j_mat = i_range.unsqueeze(1) + offsets.to(device).unsqueeze(0)  # (L, b)
        valid = (j_mat >= 0) & (j_mat < L)

        # For each (i, a) gather K, V
        # scores: (B, H, L, b)
        scores = torch.zeros(batch, H, L, b, device=device, dtype=h.dtype)

        for a_idx in range(b):
            mask_a = valid[:, a_idx]           # (L,)
            if mask_a.sum() == 0:
                continue
            j_pos = j_mat[mask_a, a_idx]       # valid j positions
            i_pos = i_range[mask_a]            # corresponding i positions

            q_sel = Q[:, :, i_pos, :]          # (B, H, n_valid, d_h)
            k_sel = K[:, :, j_pos, :]          # (B, H, n_valid, d_h)
            v_sel = V[:, :, j_pos, :]

            # Attention scores for this digit a
            s_a = (q_sel * k_sel).sum(dim=-1) / math.sqrt(d_h)  # (B, H, n_valid)

            # Add relative bias rho_{h,a}
            s_a = s_a + self.rho[:, a_idx].view(1, H, 1)

            # Add compiler prior
            lam_exp = self.lam.view(1, H, 1)   # (1, H, 1)
            pi_a = compiler_dists[:, a_idx]     # (B,)
            pi_a = pi_a.view(batch, 1, 1)      # (B, 1, 1)
            s_a = s_a + lam_exp * (pi_a + self.eps).log()

            # Scatter back
            scores[:, :, i_pos, a_idx] = s_a

        # Mask invalid positions
        scores = scores.masked_fill(~valid.unsqueeze(0).unsqueeze(0), -1e9)

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, L, b)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        output = torch.zeros_like(Q)
        for a_idx in range(b):
            mask_a = valid[:, a_idx]
            if mask_a.sum() == 0:
                continue
            j_pos = j_mat[mask_a, a_idx]
            i_pos = i_range[mask_a]
            w_a = attn_weights[:, :, i_pos, a_idx]  # (B, H, n_valid)
            v_sel = V[:, :, j_pos, :]               # (B, H, n_valid, d_h)
            out_a = w_a.unsqueeze(-1) * v_sel       # (B, H, n_valid, d_h)
            output[:, :, i_pos, :] += out_a

        output = output.permute(0, 2, 1, 3).contiguous().view(batch, L, -1)
        return self.W_O(output)


class QMRLocalHead(nn.Module):
    """
    Local content head: dense attention within a sliding window.
    Complements the routing head by capturing local context that the
    sparse routing graph cannot reach in one hop.
    """

    def __init__(self, d_model: int, num_heads: int, window_size: int = 128,
                 dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_h = d_model // num_heads
        self.window_size = window_size

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, causal: bool = False) -> torch.Tensor:
        """
        Full local attention within a sliding window.
        For efficiency on long sequences, we use a banded (block-diagonal)
        attention mask with bandwidth = window_size.
        """
        batch, L, _ = h.shape
        H = self.num_heads
        d_h = self.d_h
        device = h.device

        Q = self.W_Q(h).view(batch, L, H, d_h).permute(0, 2, 1, 3)
        K = self.W_K(h).view(batch, L, H, d_h).permute(0, 2, 1, 3)
        V = self.W_V(h).view(batch, L, H, d_h).permute(0, 2, 1, 3)

        # Compute full QK^T then mask to sliding window
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_h)  # (B,H,L,L)

        # Sliding window mask
        band_mask = torch.ones(L, L, device=device, dtype=torch.bool)
        band_mask = torch.triu(band_mask, -self.window_size) & \
                   torch.tril(band_mask, self.window_size)
        if causal:
            band_mask = torch.tril(band_mask)

        scores = scores.masked_fill(~band_mask.unsqueeze(0).unsqueeze(0), -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, V)  # (B, H, L, d_h)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch, L, -1)
        return self.W_O(output)


class QMRTransformerBlock(nn.Module):
    """
    Complete QMR-Transformer block (paper §3.4, Eq. 2–3).

    Update rule (QMR-Core+):
        h_i^{l+1} = h_i^l
                  + g_l^r * o_route^l(i)
                  + g_l^c * o_local^l(i)
                  + FFN_l(h_i^l)

    where g_l^r, g_l^c ∈ [0, 1] are learnable scalar gates per layer.
    """

    def __init__(
        self,
        d_model: int,
        num_routing_heads: int,
        num_local_heads: int,
        layer_idx: int,
        window_size: int = 128,
        ffn_hidden_mult: float = 4.0,
        dropout: float = 0.0,
        max_b: int = 64,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        # Routing head
        self.routing_head = QMRRoutingHead(
            d_model, num_routing_heads, layer_idx, max_b, dropout
        )

        # Local content head
        self.local_head = QMRLocalHead(
            d_model, num_local_heads, window_size, dropout
        )

        # Layer-wise scalar gates
        self.gate_route = nn.Parameter(torch.tensor(0.5))
        self.gate_local = nn.Parameter(torch.tensor(0.5))

        # FFN
        ffn_hidden = int(d_model * ffn_hidden_mult)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(
        self,
        h: torch.Tensor,
        compiler_dists: List[torch.Tensor],  # one per layer
        offsets_per_layer: List[torch.Tensor],
        B: List[int],
        L: int,
        D: int,
        causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            h: (batch, L, d_model)
            compiler_dists: list of D tensors, each (batch, b_l)
            offsets_per_layer: list of D tensors, each (b_l,)
            B: cumulative scales [B_0, ..., B_D]
            L, D: sequence length and depth
        """
        l_idx = self.layer_idx
        assert l_idx < D

        # Pre-norm
        h_norm = self.ln1(h)

        # Routing output
        route_out = self.routing_head(
            h_norm,
            compiler_dists[l_idx],
            offsets_per_layer[l_idx],
            B[l_idx],
            causal,
        )

        # Local content output
        local_out = self.local_head(h_norm, causal)

        # Gated combination + residual
        g_r = torch.sigmoid(self.gate_route)
        g_c = torch.sigmoid(self.gate_local)

        h = h + g_r * route_out + g_c * local_out

        # FFN
        h = h + self.ffn(self.ln2(h))

        return h


if __name__ == "__main__":
    """
    Minimal integration test: one QMR block on a short sequence.
    """
    BATCH, L, D_MODEL, D, LAYERS = 2, 256, 128, 4, 2

    gen = MixedRadixGraphGenerator()
    radices = gen.compute_radices(L, D)
    offsets = [torch.arange(b, dtype=torch.long) * gen.compute_cumulative_scales(radices)[l] * -1
               for l, b in enumerate(radices)]
    B = gen.compute_cumulative_scales(radices)

    # Dummy compiler distributions (uniform)
    compiler_dists = [
        torch.ones(BATCH, b) / b for b in radices
    ]

    block = QMRTransformerBlock(
        d_model=D_MODEL,
        num_routing_heads=4,
        num_local_heads=4,
        layer_idx=0,
    )

    h = torch.randn(BATCH, L, D_MODEL)
    h_out = block(h, compiler_dists, offsets, B, L, D)
    assert h_out.shape == h.shape
    print(f"QMRTransformerBlock test passed: {h.shape} -> {h_out.shape}")

    # Grad test
    loss = h_out.sum()
    loss.backward()
    print("Gradient flow OK")
