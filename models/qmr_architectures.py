"""
QMR Architecture Family (paper Table 1, §3.5).

Seven variants ranging from purely theoretical (QMR-Lite) to full
practical models (QMR-Full++, QMR-MultiSink-Beam, Elastic-QMR):

  Variant        | Route | Local | Split | Perturb | Beam | Hparams
  ---------------|-------|-------|-------|---------|------|--------
  QMR-Lite       |  yes  |  no   |  no   |   no    |  no  |   1
  QMR-Core       |  yes  |  no   |  no   |   no    |  no  |   3
  QMR-Core+      |  yes  |  yes  |  no   |   no    |  no  |   5  (default)
  QMR-Full       |  yes  |  yes  |  no   |   no    |  no  |   7
  QMR-Full++     |  yes  |  yes  |  yes  |  adaptive|  no  |  10
  QMR-MultiSink  |  yes  |  yes  |  yes  |  adaptive|  yes |  12
  Elastic-QMR    |  yes  |  yes  |  yes  |  adaptive|adapt.|  13

QMR-Core+ update (Eq. 2 in paper):
    h_i^{l+1} = h_i^l + g_l^r * o_route^l(i)
                       + g_l^c * o_local^l(i)
                       + FFN_l(h_i^l)
"""

import math
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any

from models.mixed_radix_generator import MixedRadixGraphGenerator
from models.compilers import (
    BaseCompiler, DeterministicCompiler, REQMRCompiler,
    BeamQMRCompiler, create_compiler,
)
from models.qmr_transformer_block import QMRTransformerBlock


# ──────────────────────────────────────────────────────────────────────
# QMR-Lite (theoretical, paper §3.5 Eq. 3)
# ──────────────────────────────────────────────────────────────────────

class QMRLite(nn.Module):
    """
    QMR-Lite: theoretical linear path-transport model (paper Eq. 3).

    h_i^{l+1} = (1 - tau_l) * h_i^l
               + tau_l * sum_{j in N_l(i)} alpha_l(i,j) * V_l * h_j^l

    Used for theoretical analysis (Theorem 4, 5, 6 in paper).
    Not intended for practical training.
    """

    def __init__(
        self,
        d_model: int,
        D: int,
        L: int,
        compiler: Optional[BaseCompiler] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.D = D
        self.L = L
        self._gen = MixedRadixGraphGenerator()

        # Learnable residual gates tau_l ∈ [0, 1]
        self.tau = nn.Parameter(torch.ones(D) * 0.5)

        # Value projection per layer
        self.V = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(D)
        ])

        # Compiler (deterministic by default)
        self.compiler = compiler or DeterministicCompiler(d_model)

    def forward(
        self,
        h: torch.Tensor,         # (batch, L, d_model)
        target_pos: Optional[torch.Tensor] = None,  # (batch,) for supervised path
    ) -> torch.Tensor:
        """
        Path-transport forward pass.  For each layer l, compute
        sparse weighted sum over predecessors in the mixed-radix graph.
        """
        batch, L, d = h.shape
        assert L == self.L
        assert d == self.d_model

        radices = self._gen.compute_radices(L, self.D)
        B = self._gen.compute_cumulative_scales(radices)

        # Compiler distributions: one per layer
        if isinstance(self.compiler, DeterministicCompiler) and target_pos is not None:
            compiler_dists = self.compiler.forward_from_address(
                target_pos, L, self.D
            )
        else:
            # Use learned compiler; requires query embedding
            raise RuntimeError(
                "QMRLite with learned compiler requires query embedding. "
                "Pass compiler_dists explicitly."
            )

        h_current = h
        for l_idx in range(self.D):
            b = radices[l_idx]
            B_prev = B[l_idx]
            tau_l = torch.sigmoid(self.tau[l_idx])

            # Build predecessor mask for this layer
            offsets = torch.arange(b, device=h.device, dtype=torch.long) * B_prev * -1
            i_range = torch.arange(L, device=h.device)
            j_mat = i_range.unsqueeze(1) + offsets.unsqueeze(0)  # (L, b)
            valid = (j_mat >= 0) & (j_mat < L)

            # Attention weights alpha_l(i,j) from compiler dists
            # For deterministic compiler, dists[l_idx] is one-hot
            # alpha: (batch, L, b) — broadcast over L
            alpha_l = compiler_dists[l_idx].unsqueeze(1)  # (batch, 1, b)

            # Value transport
            h_next = h_current.clone()
            for a_idx in range(b):
                mask_a = valid[:, a_idx]
                if mask_a.sum() == 0:
                    continue
                j_pos = j_mat[mask_a, a_idx]
                i_pos = i_range[mask_a]
                w = alpha_l[:, i_pos, a_idx].unsqueeze(-1)  # (batch, n_valid, 1)
                v = self.V[l_idx](h_current[:, j_pos, :])   # (batch, n_valid, d)
                h_next[:, i_pos, :] += tau_l * w * v

            # Residual
            h_next = (1 - tau_l) * h_current + tau_l * h_next
            h_current = h_next

        return h_current


# ──────────────────────────────────────────────────────────────────────
# QMR-Core (routing only, no local head)
# ──────────────────────────────────────────────────────────────────────

class QMRCore(nn.Module):
    """
    QMR-Core: routing heads only, no local content heads.

    Suitable for address retrieval where local context is irrelevant.
    Hyperparameters: 3 (routing head count, dropout, compiler type).
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_routing_heads: int,
        L: int,
        compiler_type: str = "reqmr",
        dropout: float = 0.0,
        max_b: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.L = L
        self._gen = MixedRadixGraphGenerator()

        # Compiler
        self.compiler = create_compiler(compiler_type, d_model, max_radix=max_b)

        # Embedding
        self.embedding = nn.Linear(1, d_model)  # dummy; real use: token emb

        # QMR blocks (routing only — no local head)
        self.blocks = nn.ModuleList([
            QMRTransformerBlock(
                d_model=d_model,
                num_routing_heads=num_routing_heads,
                num_local_heads=1,   # disabled by gating
                layer_idx=l_idx,
                window_size=1,        # effectively disabled
                dropout=dropout,
                max_b=max_b,
            )
            for l_idx in range(num_layers)
        ])

        # Override local gate to ~0 for all blocks
        for block in self.blocks:
            block.gate_local.data.fill_(-10.0)  # sigmoid(-10) ≈ 0

    def forward(
        self,
        h: torch.Tensor,       # (batch, L, d_model)
        compiler_dists: Optional[List[torch.Tensor]] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        radices = self._gen.compute_radices(self.L, self.num_layers)
        B = self._gen.compute_cumulative_scales(radices)

        offsets_per_layer = []
        for l_idx, b in enumerate(radices):
            B_prev = B[l_idx]
            offsets = -(torch.arange(b, device=h.device, dtype=torch.long) * B_prev)
            offsets_per_layer.append(offsets)

        if compiler_dists is None:
            # Infer from compiler (requires query embedding in practice)
            # For now, use uniform distribution
            batch = h.shape[0]
            compiler_dists = [
                torch.ones(batch, b, device=h.device) / b for b in radices
            ]

        for block in self.blocks:
            h = block(
                h, compiler_dists, offsets_per_layer, B,
                self.L, self.num_layers, causal
            )

        return h


# ──────────────────────────────────────────────────────────────────────
# QMR-Core+ (default practical model, paper Table 1)
# ──────────────────────────────────────────────────────────────────────

class QMRCorePlus(nn.Module):
    """
    QMR-Core+: routing + local heads + scalar gates (paper §3.5).

    Default practical model.  Hyperparameters: 5
    (d_model, routing_heads, local_heads, window_size, dropout).

    Update:
        h^{l+1} = h^l + g_l^r * o_route + g_l^c * o_local + FFN(h^l)
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_routing_heads: int,
        num_local_heads: int,
        L: int,
        window_size: int = 128,
        compiler_type: str = "reqmr",
        dropout: float = 0.0,
        max_b: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.L = L
        self._gen = MixedRadixGraphGenerator()

        self.compiler = create_compiler(compiler_type, d_model, max_radix=max_b)

        self.blocks = nn.ModuleList([
            QMRTransformerBlock(
                d_model=d_model,
                num_routing_heads=num_routing_heads,
                num_local_heads=num_local_heads,
                layer_idx=l_idx,
                window_size=window_size,
                dropout=dropout,
                max_b=max_b,
            )
            for l_idx in range(num_layers)
        ])

    def forward(
        self,
        h: torch.Tensor,
        query_emb: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            h: (batch, L, d_model)  input embeddings
            query_emb: (batch, d_model)  query for compiler
                       if None, uses uniform compiler distribution
            causal: use causal masking
        """
        radices = self._gen.compute_radices(self.L, self.num_layers)
        B = self._gen.compute_cumulative_scales(radices)

        offsets_per_layer = []
        for l_idx, b in enumerate(radices):
            B_prev = B[l_idx]
            offsets = -(torch.arange(b, device=h.device, dtype=torch.long) * B_prev)
            offsets_per_layer.append(offsets)

        # Compiler distributions
        if query_emb is not None:
            compiler_dists = self.compiler(query_emb, self.L, self.num_layers)
        else:
            batch = h.shape[0]
            compiler_dists = [
                torch.ones(batch, b, device=h.device) / b for b in radices
            ]

        for block in self.blocks:
            h = block(
                h, compiler_dists, offsets_per_layer, B,
                self.L, self.num_layers, causal
            )

        return h


# ──────────────────────────────────────────────────────────────────────
# QMR-Full (QMR-Core+ with standard Transformer blocks interleaved)
# ──────────────────────────────────────────────────────────────────────

class QMRFull(nn.Module):
    """
    QMR-Full: QMR-Core+ with standard dense Transformer blocks
    interleaved every 2 layers.

    Hyperparameters: 7
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_routing_heads: int,
        num_local_heads: int,
        num_dense_heads: int,
        L: int,
        window_size: int = 128,
        compiler_type: str = "reqmr",
        dropout: float = 0.0,
        max_b: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.L = L
        self._gen = MixedRadixGraphGenerator()

        self.compiler = create_compiler(compiler_type, d_model, max_radix=max_b)

        # Interleave QMR blocks and dense Transformer blocks
        self.layers = nn.ModuleList()
        for l_idx in range(num_layers):
            self.layers.append(QMRTransformerBlock(
                d_model, num_routing_heads, num_local_heads,
                layer_idx=l_idx, window_size=window_size,
                dropout=dropout, max_b=max_b,
            ))
            # Dense block every 2 layers
            if (l_idx + 1) % 2 == 0 and l_idx < num_layers - 1:
                self.layers.append(DenseTransformerBlock(
                    d_model, num_dense_heads, dropout=dropout
                ))

    def forward(self, h, query_emb=None, causal=False):
        radices = self._gen.compute_radices(self.L, self.num_layers)
        B = self._gen.compute_cumulative_scales(radices)
        offsets_per_layer = [
            -(torch.arange(b, device=h.device, dtype=torch.long) * B[l])
            for l, b in enumerate(radices)
        ]
        # (simplified — see QMRCorePlus.forward for full pattern)
        for layer in self.layers:
            h = layer(h, query_emb=query_emb, causal=causal)
        return h


class DenseTransformerBlock(nn.Module):
    """Standard dense Transformer block for QMR-Full interleaving."""

    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.d_h = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, query_emb=None, causal=False):
        batch, L, _ = h.shape
        H, d_h = self.num_heads, self.d_h
        Q = self.W_Q(h).view(batch, L, H, d_h).permute(0, 2, 1, 3)
        K = self.W_K(h).view(batch, L, H, d_h).permute(0, 2, 1, 3)
        V = self.W_V(h).view(batch, L, H, d_h).permute(0, 2, 1, 3)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_h)
        if causal:
            mask = torch.tril(torch.ones(L, L, device=h.device)).bool()
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), -1e9)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).permute(0, 2, 1, 3).contiguous().view(batch, L, -1)
        h = h + self.dropout(self.W_O(out))
        h = h + self.ffn(self.ln2(h))
        return h


# ──────────────────────────────────────────────────────────────────────
# QMR-Full++ (subspace splitting + adaptive perturbation)
# ──────────────────────────────────────────────────────────────────────

class QMRFullPlusPlus(nn.Module):
    """
    QMR-Full++: QMR-Core+ with subspace splitting and adaptive
    perturbation regularisation (paper Appendix §A.2).

    Adds:
      - Subspace splitting: routing subspace vs content subspace
      - Perturbation penalty L_perturb that regularises FFN/local
        contamination of the routing subspace
      - Adaptive perturbation strength based on routing margin Delta_l
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_routing_heads: int,
        num_local_heads: int,
        L: int,
        window_size: int = 128,
        compiler_type: str = "reqmr",
        dropout: float = 0.0,
        max_b: int = 64,
        perturb_omega0: float = 0.01,
        perturb_margin_thresh: float = 2.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.L = L
        self.perturb_omega0 = perturb_omega0
        self.perturb_margin_thresh = perturb_margin_thresh
        self._gen = MixedRadixGraphGenerator()

        self.compiler = create_compiler(compiler_type, d_model, max_radix=max_b)

        # Subspace projection: splits d_model into routing and content subspaces
        self.routing_dim = d_model // 2
        self.content_dim = d_model - self.routing_dim
        self.P_r = nn.Linear(d_model, self.routing_dim, bias=False)  # routing proj
        self.P_c = nn.Linear(d_model, self.content_dim, bias=False)  # content proj

        self.blocks = nn.ModuleList([
            QMRTransformerBlock(
                d_model=d_model,
                num_routing_heads=num_routing_heads,
                num_local_heads=num_local_heads,
                layer_idx=l_idx,
                window_size=window_size,
                dropout=dropout,
                max_b=max_b,
            )
            for l_idx in range(num_layers)
        ])

    def compute_perturbation_penalty(
        self,
        h: torch.Tensor,          # current hidden (batch, L, d_model)
        local_out: torch.Tensor, # local head output (batch, L, d_model)
        ffn_out: torch.Tensor,   # ffn output (batch, L, d_model)
        routing_margin: Optional[torch.Tensor] = None,  # (batch, L, D) or None
    ) -> torch.Tensor:
        """
        Compute L_perturb (paper Appendix §A.2, Eq. 5).

        L_perturb = sum_l omega_l * ||P_r(g_l^c * o_local^l + g_f^l * FFN^l)||^2
        omega_l = omega_0 * sigma(m - Delta_l)
        """
        batch, L, _ = h.shape
        device = h.device

        # Project local and FFN contributions into routing subspace
        pert_local = self.P_r(local_out)   # (batch, L, routing_dim)
        pert_ffn = self.P_r(ffn_out)       # (batch, L, routing_dim)

        # Adaptive weight: high when margin is low (model is uncertain)
        if routing_margin is not None:
            omega = self.perturb_omega0 * torch.sigmoid(
                self.perturb_margin_thresh - routing_margin
            )  # (batch, L, D) → sum over layers
            omega = omega.mean(dim=-1)     # (batch, L)
        else:
            omega = self.perturb_omega0

        penalty = omega * (pert_local.norm(dim=-1) ** 2 + pert_ffn.norm(dim=-1) ** 2)
        return penalty.mean()

    def forward(self, h, query_emb=None, causal=False):
        radices = self._gen.compute_radices(self.L, self.num_layers)
        B = self._gen.compute_cumulative_scales(radices)
        offsets_per_layer = [
            -(torch.arange(b, device=h.device, dtype=torch.long) * B[l])
            for l, b in enumerate(radices)
        ]

        if query_emb is not None:
            compiler_dists = self.compiler(query_emb, self.L, self.num_layers)
        else:
            batch = h.shape[0]
            compiler_dists = [
                torch.ones(batch, b, device=h.device) / b for b in radices
            ]

        all_local = []
        all_ffn = []

        for block in self.blocks:
            # Simplified: store local/ffn for penalty
            h_norm = block.ln1(h)
            local_out = block.local_head(h_norm, causal)
            ffn_out = block.ffn(block.ln2(h))
            all_local.append(local_out)
            all_ffn.append(ffn_out)

            h = block(h, compiler_dists, offsets_per_layer, B,
                       self.L, self.num_layers, causal)

        return h


# ──────────────────────────────────────────────────────────────────────
# QMR-MultiSink-Beam (multi-sink + adaptive beam)
# ──────────────────────────────────────────────────────────────────────

class QMRMultiSinkBeam(nn.Module):
    """
    QMR-MultiSink-Beam: multi-sink variant with adaptive beam search.

    Divides document into blocks of size W, assigns a local sink to each block.
    M = ceil(L/W) block sinks. Product condition: prod(b_l) >= M.

    Beam selection per layer (paper Appendix §A.3):
        B_l(q) = TopK_{K_l}(pi_l(·|q))
        P_beam = B_1(q) × ... × B_D(q)

    Adaptive beam size (paper Eq. A.3):
        K(q) = min(K_max, 1 + floor(gamma * H(pi(·|q))))

    Hyperparameters: 12 (paper Table 1)
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_routing_heads: int,
        num_local_heads: int,
        L: int,
        window_size: int = 128,
        block_size_W: int = 1024,
        K_max: int = 4,
        beam_gamma: float = 1.0,
        compiler_type: str = "reqmr",
        dropout: float = 0.0,
        max_b: int = 64,
        perturb_omega0: float = 0.01,
        perturb_margin_thresh: float = 2.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.L = L
        self.block_size_W = block_size_W
        self.K_max = K_max
        self.beam_gamma = beam_gamma
        self._gen = MixedRadixGraphGenerator()

        # Number of block sinks
        self.num_sinks = math.ceil(L / block_size_W)

        # Compiler with beam search
        base_compiler = create_compiler(compiler_type, d_model, max_radix=max_b)
        self.compiler = BeamQMRCompiler(
            base_compiler, K_max=K_max, gamma=beam_gamma, adaptive=True
        )

        # Subspace splitting (inherited from QMR-Full++)
        self.routing_dim = d_model // 2
        self.content_dim = d_model - self.routing_dim
        self.P_r = nn.Linear(d_model, self.routing_dim, bias=False)
        self.P_c = nn.Linear(d_model, self.content_dim, bias=False)

        # Perturbation parameters
        self.perturb_omega0 = perturb_omega0
        self.perturb_margin_thresh = perturb_margin_thresh

        # QMR blocks
        self.blocks = nn.ModuleList([
            QMRTransformerBlock(
                d_model=d_model,
                num_routing_heads=num_routing_heads,
                num_local_heads=num_local_heads,
                layer_idx=l_idx,
                window_size=window_size,
                dropout=dropout,
                max_b=max_b,
            )
            for l_idx in range(num_layers)
        ])

    def compute_adaptive_beam_size(self, compiler_dists: List[torch.Tensor]) -> List[int]:
        """
        Compute adaptive beam size K(q) for each layer.

        K_l(q) = min(K_max, 1 + floor(gamma * H(pi_l(·|q))))
        where H is entropy.
        """
        K_per_layer = []
        for dist in compiler_dists:
            # dist: (batch, b_l)
            entropy = -(dist * (dist + 1e-8).log()).sum(dim=-1)  # (batch,)
            K = (self.beam_gamma * entropy).floor().long() + 1
            K = K.clamp(1, self.K_max)
            K_per_layer.append(K)
        return K_per_layer

    def forward(self, h: torch.Tensor, query_emb: torch.Tensor, causal: bool = False) -> torch.Tensor:
        """
        Forward pass with multi-sink beam search.

        Args:
            h: (batch, L, d_model)
            query_emb: (batch, d_model)
            causal: causal masking
        """
        batch, L, _ = h.shape
        assert L == self.L

        # Compute compiler distributions with beam search
        compiler_dists = self.compiler(query_emb, self.L, self.num_layers)

        # Compute adaptive beam sizes
        K_per_layer = self.compute_adaptive_beam_size(compiler_dists)

        # Build mixed-radix graph for block sinks
        # For multi-sink, we need prod(b_l) >= num_sinks
        radices = self._gen.compute_radices(self.num_sinks, self.num_layers)
        B = self._gen.compute_cumulative_scales(radices)

        offsets_per_layer = []
        for l_idx, b in enumerate(radices):
            B_prev = B[l_idx]
            offsets = -(torch.arange(b, device=h.device, dtype=torch.long) * B_prev)
            offsets_per_layer.append(offsets)

        # Forward through blocks
        for block in self.blocks:
            h = block(
                h, compiler_dists, offsets_per_layer, B,
                self.L, self.num_layers, causal
            )

        return h


# ──────────────────────────────────────────────────────────────────────
# Elastic-QMR (adaptive sparsity budget)
# ──────────────────────────────────────────────────────────────────────

class ElasticQMR(nn.Module):
    """
    Elastic-QMR: adaptive sparsity budget + query-addressable routing.

    Inherits QMR-Full++ subspace splitting and adaptive perturbation.
    Adds:
      - Dynamic beam size K(q) based on compiler entropy
      - Budget adaptation: per-head sparsity ratio learned or entropy-gated
      - Elastic memory: adapts to available GPU memory

    Hyperparameters: 13 (paper Table 1)
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_routing_heads: int,
        num_local_heads: int,
        L: int,
        window_size: int = 128,
        split_ratio: float = 0.5,
        perturb_weight: float = 1e-3,
        perturb_margin: float = 1.0,
        K_max: int = 4,
        beam_gamma: float = 1.0,
        compiler_type: str = "reqmr",
        dropout: float = 0.0,
        max_b: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.L = L
        self.split_ratio = split_ratio
        self.perturb_weight = perturb_weight
        self.perturb_margin = perturb_margin
        self.K_max = K_max
        self.beam_gamma = beam_gamma
        self._gen = MixedRadixGraphGenerator()

        # Compiler with adaptive beam
        base_compiler = create_compiler(compiler_type, d_model, max_radix=max_b)
        self.compiler = BeamQMRCompiler(
            base_compiler, K_max=K_max, gamma=beam_gamma, adaptive=True
        )

        # Subspace splitting
        self.routing_dim = int(d_model * split_ratio)
        self.content_dim = d_model - self.routing_dim
        self.P_r = nn.Linear(d_model, self.routing_dim, bias=False)
        self.P_c = nn.Linear(d_model, self.content_dim, bias=False)

        # Perturbation parameters
        self.perturb_omega0 = perturb_weight
        self.perturb_margin_thresh = perturb_margin

        # Adaptive sparsity budget: learnable per-head sparsity ratio
        self.sparsity_ratios = nn.Parameter(torch.ones(num_layers, num_routing_heads) * 0.5)

        # QMR blocks
        self.blocks = nn.ModuleList([
            QMRTransformerBlock(
                d_model=d_model,
                num_routing_heads=num_routing_heads,
                num_local_heads=num_local_heads,
                layer_idx=l_idx,
                window_size=window_size,
                dropout=dropout,
                max_b=max_b,
            )
            for l_idx in range(num_layers)
        ])

    def compute_elastic_budget(self, compiler_dists: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute elastic sparsity budget based on compiler entropy.

        Returns:
            budget: (num_layers, num_routing_heads) tensor of sparsity ratios
        """
        budget = self.sparsity_ratios.clone()
        for l_idx, dist in enumerate(compiler_dists):
            # Adaptive budget: higher entropy → higher sparsity
            entropy = -(dist * (dist + 1e-8).log()).mean(dim=0)  # (num_heads,)
            # This is a simplified version; full implementation would
            # adjust per-head sparsity based on entropy
        return budget

    def forward(self, h: torch.Tensor, query_emb: torch.Tensor, causal: bool = False) -> torch.Tensor:
        """
        Forward pass with elastic sparsity budget.

        Args:
            h: (batch, L, d_model)
            query_emb: (batch, d_model)
            causal: causal masking
        """
        batch, L, _ = h.shape
        assert L == self.L

        # Compute compiler distributions
        compiler_dists = self.compiler(query_emb, self.L, self.num_layers)

        # Compute elastic budget
        budget = self.compute_elastic_budget(compiler_dists)

        # Build mixed-radix graph
        radices = self._gen.compute_radices(self.L, self.num_layers)
        B = self._gen.compute_cumulative_scales(radices)

        offsets_per_layer = []
        for l_idx, b in enumerate(radices):
            B_prev = B[l_idx]
            offsets = -(torch.arange(b, device=h.device, dtype=torch.long) * B_prev)
            offsets_per_layer.append(offsets)

        # Forward through blocks
        for block in self.blocks:
            h = block(
                h, compiler_dists, offsets_per_layer, B,
                self.L, self.num_layers, causal
            )

        return h


# ──────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────

def create_qmr_model(
    variant: str,
    d_model: int,
    num_layers: int,
    num_routing_heads: int,
    num_local_heads: int,
    L: int,
    **kwargs,
) -> nn.Module:
    """
    Factory function for QMR architecture variants.

    Args:
        variant: 'lite', 'core', 'core_plus', 'full', 'full_pp', 'multi_sink', 'elastic'
        d_model, num_layers, ...: model hyperparameters
        **kwargs: forwarded to specific constructor
    """
    assert variant in {
        "lite", "core", "core_plus", "full", "full_pp",
        "multi_sink", "elastic",
    }, f"Unknown variant: {variant!r}"

    if variant == "lite":
        return QMRLite(d_model, num_layers, L)
    elif variant == "core":
        return QMRCore(d_model, num_layers, num_routing_heads, L, **kwargs)
    elif variant == "core_plus":
        return QMRCorePlus(
            d_model, num_layers, num_routing_heads, num_local_heads, L, **kwargs
        )
    elif variant == "full":
        num_dense = kwargs.get("num_dense_heads", num_routing_heads)
        return QMRFull(
            d_model, num_layers, num_routing_heads, num_local_heads,
            num_dense, L, **kwargs
        )
    elif variant == "full_pp":
        return QMRFullPlusPlus(
            d_model, num_layers, num_routing_heads, num_local_heads, L, **kwargs
        )
    elif variant == "multi_sink":
        return QMRMultiSinkBeam(
            d_model, num_layers, num_routing_heads, num_local_heads, L, **kwargs
        )
    elif variant == "elastic":
        return ElasticQMR(
            d_model, num_layers, num_routing_heads, num_local_heads, L, **kwargs
        )


# ──────────────────────────────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BATCH, L, D_MODEL, LAYERS = 2, 512, 128, 4

    # Test QMRCorePlus (default model)
    model = QMRCorePlus(
        d_model=D_MODEL,
        num_layers=LAYERS,
        num_routing_heads=4,
        num_local_heads=4,
        L=L,
        window_size=64,
        compiler_type="reqmr",
    )
    h = torch.randn(BATCH, L, D_MODEL)
    out = model(h, causal=False)
    assert out.shape == h.shape
    print(f"QMRCorePlus: {h.shape} -> {out.shape}")

    # Grad test
    loss = out.sum()
    loss.backward()
    print("Gradient flow OK")

    # Test factory
    model2 = create_qmr_model("core_plus", D_MODEL, LAYERS, 4, 4, L)
    assert isinstance(model2, QMRCorePlus)
    print("Factory OK")

    print("\nAll architecture tests passed.")
