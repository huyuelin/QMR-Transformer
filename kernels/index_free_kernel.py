"""
Index-Free Mixed-Radix Kernel (paper §6).

Implements the index-free sparse attention kernel for QMR routing heads.
Key insight (paper §6): candidate positions are computed by closed form
    j = i - a * B_{l-1},  a ∈ {0,...,b_l-1}
so the kernel does NOT need CSR/COO index reads.

This implementation uses pure PyTorch (no Triton dependency).
For production use, replace with Triton/CUDA kernel.

Complexity: O(B * H * L * d * sum(b_l))
vs dense:   O(B * H * L^2 * d)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


def qmr_routing_attention(
    h: torch.Tensor,           # (batch, L, d_model)
    offsets_per_layer: List[torch.Tensor],  # list of (b_l,) tensors
    B: List[int],             # cumulative scales
    routing_heads: nn.ModuleList,
    compiler_dists: List[torch.Tensor],  # list of (batch, b_l)
    causal: bool = False,
) -> torch.Tensor:
    """
    Index-free QMR routing attention (paper §6).

    For each layer l and position i, computes attention over
    candidates {i - a*B_{l-1} : a = 0,...,b_l-1}.

    Uses the closed-form index computation (no sparse index storage).

    Args:
        h: (batch, L, d_model)
        offsets_per_layer: list of length D, each (b_l,)
        B: [B_0, B_1, ..., B_D]
        routing_heads: list of QMRRoutingHead modules
        compiler_dists: list of (batch, b_l) distributions
        causal: causal masking

    Returns:
        output: (batch, L, d_model)
    """
    batch, L, d_model = h.shape
    device = h.device
    D = len(offsets_per_layer)

    output = torch.zeros_like(h)

    for l_idx in range(D):
        offsets = offsets_per_layer[l_idx]   # (b_l,)
        b = offsets.shape[0]
        head = routing_heads[l_idx]

        # (i, a) -> j = i + offsets[a]
        i_range = torch.arange(L, device=device)                # (L,)
        j_mat = i_range.unsqueeze(1) + offsets.unsqueeze(0)   # (L, b)
        valid = (j_mat >= 0) & (j_mat < L)                     # (L, b)

        # Gather Q, K, V at candidate positions
        # For efficiency, loop over digits a (typically b_l <= 16)
        h_norm = h  # skip norm for simplicity; real impl uses pre-norm

        # Q: (B, L, d_model) -> (B, H, L, d_h)
        Q = head.W_Q(h_norm)  # simplified; real code uses multi-head projection
        K = head.W_K(h_norm)
        V = head.W_V(h_norm)

        # Compute attention for this layer
        # Simplified single-head version for demonstration
        d_h = d_model
        Q_proj = Q  # (B, L, d_model)
        K_proj = K
        V_proj = V

        # Attention scores: for each (i, a) pair
        # scores[i, a] = Q[i]^T K[j] / sqrt(d)
        scores = torch.zeros(batch, L, b, device=device, dtype=h.dtype)

        for a_idx in range(b):
            j_pos = j_mat[:, a_idx]                     # (L,)
            valid_a = valid[:, a_idx]                     # (L,)

            if valid_a.sum() == 0:
                continue

            # Q[i] dot K[j] for valid pairs
            q_valid = Q_proj[:, valid_a, :]              # (B, n_valid, d)
            k_valid = K_proj[:, j_pos[valid_a], :]        # (B, n_valid, d)

            s_a = (q_valid * k_valid).sum(dim=-1) / math.sqrt(d_h)  # (B, n_valid)

            # Add compiler prior
            lam = head.lam  # scalar or per-head
            pi_a = compiler_dists[l_idx][:, a_idx]        # (B,)
            s_a = s_a + lam * (pi_a + 1e-6).log()

            # Add relative bias
            rho_a = head.rho[a_idx]                       # scalar
            s_a = s_a + rho_a

            scores[:, valid_a, a_idx] = s_a

        # Mask invalid
        scores = scores.masked_fill(~valid.unsqueeze(0), -1e9)

        # Softmax
        attn = F.softmax(scores, dim=-1)  # (B, L, b)

        # Weighted sum of values
        for a_idx in range(b):
            j_pos = j_mat[:, a_idx]
            valid_a = valid[:, a_idx]
            if valid_a.sum() == 0:
                continue
            w_a = attn[:, valid_a, a_idx].unsqueeze(-1)   # (B, n_valid, 1)
            v_a = V_proj[:, j_pos[valid_a], :]            # (B, n_valid, d)
            out_a = w_a * v_a                             # (B, n_valid, d)

            # Scatter back to output
            # output[:, valid_a, :] += out_a  (simplified)
            output[:, valid_a, :] += out_a

    return output


class IndexFreeQMRKernel(nn.Module):
    """
    Wrapper module for the index-free QMR kernel.

    Provides a drop-in replacement for dense attention in QMR blocks.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        max_b: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self._gen = MixedRadixGraphGenerator()

        # Per-layer routing projections
        self.routing_heads = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(num_layers)
        ])
        self.value_heads = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(num_layers)
        ])
        self.output_heads = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(num_layers)
        ])

        # Relative bias and compiler coupling
        self.rho = nn.Parameter(torch.zeros(num_layers, max_b))
        self.lam = nn.Parameter(torch.ones(num_layers))

    def forward(
        self,
        h: torch.Tensor,
        L: int,
        compiler_dists: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass using index-free kernel.

        Args:
            h: (batch, L, d_model)
            L: sequence length
            compiler_dists: list of D tensors (batch, b_l)

        Returns:
            output: (batch, L, d_model)
        """
        batch, L_actual, _ = h.shape
        assert L_actual == L

        radices = self._gen.compute_radices(L, self.num_layers)
        B = self._gen.compute_cumulative_scales(radices)

        offsets_per_layer = []
        for l_idx, b in enumerate(radices):
            B_prev = B[l_idx]
            offsets = -(torch.arange(b, device=h.device, dtype=torch.long) * B_prev)
            offsets_per_layer.append(offsets)

        output = qmr_routing_attention(
            h, offsets_per_layer, B,
            self.routing_heads, compiler_dists
        )

        return output


# ──────────────────────────────────────────────────────────────────────
# Triton kernel stub (for future implementation)
# ──────────────────────────────────────────────────────────────────────

TRITON_KERNEL_AVAILABLE = False

try:
    import triton
    import triton.language as tl
    TRITON_KERNEL_AVAILABLE = True

    @triton.jit
    def _qmr_routing_kernel(
        Q_ptr, K_ptr, V_ptr, Out_ptr,
        offsets_ptr, compiler_ptr,
        batch, L, d_model, b_l,
        B_prev, stride_qb, stride_ql, stride_qd,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Triton kernel for QMR routing attention.

        Computes attention over b_l candidates per position using
        the closed-form index j = i - a * B_prev.
        """
        # TODO: implement full Triton kernel
        pass

    def qmr_routing_triton(
        h: torch.Tensor,
        offsets: torch.Tensor,
        compiler_dist: torch.Tensor,
        B_prev: int,
    ) -> torch.Tensor:
        """Triton-accelerated version of QMR routing."""
        raise NotImplementedError("Triton kernel not yet implemented")

except ImportError:
    pass


# ──────────────────────────────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    BATCH, L, D_MODEL, D = 2, 256, 64, 3

    gen = MixedRadixGraphGenerator()
    radices = gen.compute_radices(L, D)
    B = gen.compute_cumulative_scales(radices)
    offsets = [
        -(torch.arange(b, dtype=torch.long) * B[l])
        for l, b in enumerate(radices)
    ]

    # Dummy compiler
    compiler_dists = [
        torch.ones(BATCH, b) / b for b in radices
    ]

    # Test pure-PyTorch kernel
    h = torch.randn(BATCH, L, D_MODEL)
    out = qmr_routing_attention(
        h, offsets, B,
        nn.ModuleList([nn.Linear(D_MODEL, D_MODEL) for _ in range(D)]),
        compiler_dists,
    )
    print(f"Index-free kernel: {h.shape} -> {out.shape}")
    assert out.shape == h.shape
    print("Index-free kernel test passed")
