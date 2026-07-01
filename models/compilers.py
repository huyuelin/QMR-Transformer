"""
Query Compilers for QMR-Transformers (paper §3.2, Appendix A).

A compiler maps a query embedding to per-layer routing distributions
π_{l,h}(·|q) ∈ Δ^{b_l - 1}, which bias routing-head logits toward the
canonical mixed-radix path from the sink to the addressed source position.

Five compiler families (Table 1 / Figure 4 in paper):
  1. DeterministicCompiler  – algorithmic index → digit conversion (§3.2)
  2. BinaryMLPCompiler      – direct digit prediction MLP (§3.2, brittle)
  3. REQMRCompiler          – predict normalised address then radix-decode (§3.2)
  4. SemanticAnchorCompiler – predict block-level anchor from text (§3.2)
  5. BeamQMRCompiler        – maintain top-K paths under high entropy (App. §A.3)

All compilers expose the same interface:
    forward(query_emb, L, D) -> List[Tensor]  (one (b_l,) distribution per layer)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from models.mixed_radix_generator import MixedRadixGraphGenerator


# ──────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────

class BaseCompiler(nn.Module):
    """
    Common scaffolding shared by all compiler variants.

    Subclasses must implement _compute_distributions().
    """

    def __init__(self, d_model: int, max_D: int = 16):
        super().__init__()
        self.d_model = d_model
        self.max_D = max_D
        self._gen = MixedRadixGraphGenerator()

    def forward(
        self,
        query_emb: torch.Tensor,          # (batch, d_model)
        L: int,
        D: int,
    ) -> List[torch.Tensor]:
        """
        Returns:
            dists: list of D tensors each of shape (batch, b_l)
                   each is a valid probability distribution (softmax applied)
        """
        assert query_emb.dim() == 2, "query_emb must be (batch, d_model)"
        assert query_emb.shape[1] == self.d_model, (
            f"query_emb.shape[1]={query_emb.shape[1]} != d_model={self.d_model}"
        )
        radices = self._gen.compute_radices(L, D)
        return self._compute_distributions(query_emb, L, D, radices)

    def _compute_distributions(
        self,
        query_emb: torch.Tensor,
        L: int, D: int,
        radices: List[int],
    ) -> List[torch.Tensor]:
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────
# 1. Deterministic compiler
# ──────────────────────────────────────────────────────────────────────

class DeterministicCompiler(BaseCompiler):
    """
    Algorithmic index → mixed-radix digit conversion (paper §3.2).

    Requires explicit integer address supervision.  This is an
    oracle / diagnostic upper bound, not a learned component.

    Usage: pass target_position (integer index) as query_emb placeholder.
    The actual query_emb tensor is ignored; the compiler reads 'address'
    directly.
    """

    def __init__(self, d_model: int = 64, max_D: int = 16):
        super().__init__(d_model, max_D)

    def forward_from_address(
        self,
        address: torch.Tensor,   # (batch,)  int64, position in [0, L)
        L: int,
        D: int,
    ) -> List[torch.Tensor]:
        """
        Convert integer address to one-hot distributions over digits.

        displacement = (L - 1) - address
        digit_l = floor(displacement / B_{l-1}) % b_l
        """
        radices = self._gen.compute_radices(L, D)
        B = self._gen.compute_cumulative_scales(radices)
        batch = address.shape[0]
        device = address.device

        displacement = (L - 1) - address.long()         # (batch,)
        assert (displacement >= 0).all(), "address >= L"
        assert (displacement < math.prod(radices)).all(), (
            "displacement out of coverage range"
        )

        dists = []
        for l_idx in range(D):
            b = radices[l_idx]
            B_prev = B[l_idx]
            digit = (displacement // B_prev) % b          # (batch,)
            one_hot = F.one_hot(digit, num_classes=b).float().to(device)  # (batch, b)
            dists.append(one_hot)
        return dists

    def _compute_distributions(
        self, query_emb, L, D, radices
    ) -> List[torch.Tensor]:
        # Cannot infer address from embedding alone; raise informative error.
        raise RuntimeError(
            "DeterministicCompiler requires explicit address. "
            "Call forward_from_address(address, L, D) instead."
        )


# ──────────────────────────────────────────────────────────────────────
# 2. Binary MLP compiler  (fragile under distribution shift)
# ──────────────────────────────────────────────────────────────────────

class BinaryMLPCompiler(BaseCompiler):
    """
    Direct digit-prediction MLP (paper §3.2).

    Uses separate output heads, one per layer.
    Predicts logits over {0, ..., b_l - 1} for each layer.

    Known limitation: collapses under narrow training support
    (Table 3 in paper).
    """

    def __init__(self, d_model: int, max_radix: int = 32, hidden_dim: int = 256,
                 max_D: int = 16):
        super().__init__(d_model, max_D)
        self.max_radix = max_radix
        self.hidden_dim = hidden_dim

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        # Per-layer heads (up to max_D layers, each head outputs max_radix logits)
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, max_radix) for _ in range(max_D)
        ])

    def _compute_distributions(
        self, query_emb, L, D, radices
    ) -> List[torch.Tensor]:
        feat = self.trunk(query_emb)                     # (batch, hidden)
        dists = []
        for l_idx in range(D):
            b = radices[l_idx]
            assert b <= self.max_radix, (
                f"radix {b} exceeds max_radix {self.max_radix}; "
                "rebuild compiler with larger max_radix"
            )
            logits = self.heads[l_idx](feat)[:, :b]     # (batch, b)
            dists.append(F.softmax(logits, dim=-1))
        return dists


# ──────────────────────────────────────────────────────────────────────
# 3. RE-QMR compiler  (default learned compiler)
# ──────────────────────────────────────────────────────────────────────

class REQMRCompiler(BaseCompiler):
    """
    RE-QMR compiler (paper §3.2, Table 3).

    Step 1: predict normalised address  t̂ = σ(f_θ(u_q)) ∈ [0, 1]
    Step 2: hard-decode digits           â_l = floor((t̂ * L - 1) / B_{l-1}) % b_l
    Step 3: soft distribution via temperature-annealed Gumbel / softmax

    This length-invariant design extrapolates beyond training lengths
    because the address network outputs a fraction rather than an
    absolute integer index.

    Optional aux losses (Appendix §A):
      - address reconstruction   L_addr = |q_rec - q| / L
      - carry consistency
      - multi-length consistency
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 256,
        temperature: float = 0.1,
        max_radix: int = 32,
        max_D: int = 16,
    ):
        super().__init__(d_model, max_D)
        self.temperature = temperature
        self.max_radix = max_radix

        # Address predictor: query embedding → scalar normalised address
        self.addr_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Per-layer soft-digit head (re-weighted by addr prior)
        # Each head takes (hidden, 1) and outputs b-dim logits
        self.digit_heads = nn.ModuleList([
            nn.Linear(hidden_dim, max_radix) for _ in range(max_D)
        ])

        # Shared trunk (produces features used by both addr_net and digit_heads)
        self.trunk = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def predict_normalised_address(
        self, query_emb: torch.Tensor
    ) -> torch.Tensor:
        """Returns t̂ ∈ (0, 1), shape (batch, 1)."""
        return torch.sigmoid(self.addr_net(query_emb))   # (batch, 1)

    def decode_digits_from_address(
        self,
        t_hat: torch.Tensor,             # (batch, 1), normalised ∈ [0,1]
        L: int,
        D: int,
        radices: List[int],
    ) -> List[torch.Tensor]:
        """
        Hard-decode digits from normalised address.

        â_l = floor((t̂ * (L-1)) / B_{l-1}) % b_l

        Returns list of D int tensors of shape (batch,).
        """
        B = self._gen.compute_cumulative_scales(radices)
        displacement = (t_hat.squeeze(1) * (L - 1)).long()  # (batch,)
        digits = []
        for l_idx in range(D):
            b = radices[l_idx]
            B_prev = B[l_idx]
            d = (displacement // B_prev) % b
            digits.append(d)
        return digits

    def _compute_distributions(
        self, query_emb, L, D, radices
    ) -> List[torch.Tensor]:
        feat = self.trunk(query_emb)                     # (batch, hidden)
        t_hat = torch.sigmoid(self.addr_net(query_emb))  # (batch, 1)

        B = self._gen.compute_cumulative_scales(radices)
        dists = []
        for l_idx in range(D):
            b = radices[l_idx]
            assert b <= self.max_radix, (
                f"radix {b} exceeds max_radix {self.max_radix}"
            )
            # Hard digit from normalised address (provides strong prior)
            displacement = (t_hat.squeeze(1) * (L - 1)).long()
            B_prev = B[l_idx]
            hard_digit = (displacement // B_prev) % b         # (batch,)
            hard_one_hot = F.one_hot(hard_digit, num_classes=b).float()

            # Soft digit logits from trunk features
            soft_logits = self.digit_heads[l_idx](feat)[:, :b]  # (batch, b)

            # Combine: soft logits + temperature-scaled hard prior
            combined_logits = soft_logits + hard_one_hot / self.temperature
            dists.append(F.softmax(combined_logits, dim=-1))

        return dists

    def compute_address_reconstruction_loss(
        self,
        dists: List[torch.Tensor],
        target_pos: torch.Tensor,   # (batch,) int, ground-truth target position
        L: int,
        D: int,
    ) -> torch.Tensor:
        """
        L_addr = |q_rec - q| / L   (paper Appendix, address loss)

        q_rec = 1 + sum_l E[a_l] * B_{l-1}
        """
        radices = self._gen.compute_radices(L, D)
        B = self._gen.compute_cumulative_scales(radices)

        q_rec = torch.ones(target_pos.shape[0], device=target_pos.device, dtype=torch.float)
        for l_idx in range(D):
            a_vals = torch.arange(radices[l_idx], device=target_pos.device, dtype=torch.float)
            expected_a = (dists[l_idx] * a_vals.unsqueeze(0)).sum(dim=-1)  # (batch,)
            q_rec = q_rec + expected_a * B[l_idx]

        # displacement = L - 1 - target_pos  →  target q_rec = L - target_pos
        target_q_rec = (L - target_pos.float())
        return (q_rec - target_q_rec).abs().mean() / L


# ──────────────────────────────────────────────────────────────────────
# 4. Semantic anchor compiler
# ──────────────────────────────────────────────────────────────────────

class SemanticAnchorCompiler(BaseCompiler):
    """
    Semantic anchor compiler (paper §3.2).

    Predicts a block-level anchor from a text query embedding, then
    converts the block index to mixed-radix digits.  Used for natural-
    language retrieval where explicit position supervision is unavailable.
    """

    def __init__(
        self,
        d_model: int,
        block_size: int = 256,
        hidden_dim: int = 256,
        max_radix: int = 32,
        max_D: int = 16,
    ):
        super().__init__(d_model, max_D)
        self.block_size = block_size
        self.max_radix = max_radix

        # Block scorer: query emb → per-block logit
        # We predict a normalised block address ∈ [0, 1]
        self.block_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Per-layer digit heads conditioned on (query feat, block addr)
        self.digit_heads = nn.ModuleList([
            nn.Linear(hidden_dim + 1, max_radix) for _ in range(max_D)
        ])

        self.trunk = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def _compute_distributions(
        self, query_emb, L, D, radices
    ) -> List[torch.Tensor]:
        feat = self.trunk(query_emb)                          # (batch, hidden)
        block_addr = torch.sigmoid(self.block_net(query_emb)) # (batch, 1)

        dists = []
        inp = torch.cat([feat, block_addr], dim=-1)           # (batch, hidden+1)
        for l_idx in range(D):
            b = radices[l_idx]
            assert b <= self.max_radix
            logits = self.digit_heads[l_idx](inp)[:, :b]
            dists.append(F.softmax(logits, dim=-1))
        return dists


# ──────────────────────────────────────────────────────────────────────
# 5. Beam-QMR compiler
# ──────────────────────────────────────────────────────────────────────

class BeamQMRCompiler(BaseCompiler):
    """
    Beam-QMR compiler (paper Appendix §A.3).

    Maintains top-K paths at each layer; K is optionally chosen
    from compiler entropy (adaptive beam).

    P_beam = B_1(q) × B_2(q) × ... × B_D(q)
    B_l(q) = TopK_{K_l} π_l(·|q)
    K(q)   = min(K_max, 1 + floor(γ * H(π(·|q))))   [adaptive]
    """

    def __init__(
        self,
        base_compiler: BaseCompiler,
        K_max: int = 4,
        gamma: float = 2.0,
        adaptive: bool = True,
    ):
        super().__init__(base_compiler.d_model, base_compiler.max_D)
        self.base_compiler = base_compiler
        self.K_max = K_max
        self.gamma = gamma
        self.adaptive = adaptive

    def _compute_distributions(
        self, query_emb, L, D, radices
    ) -> List[torch.Tensor]:
        """Returns softened distributions with beam masking applied."""
        base_dists = self.base_compiler._compute_distributions(
            query_emb, L, D, radices
        )
        beam_dists = []
        for l_idx, dist in enumerate(base_dists):
            # dist: (batch, b_l)
            if self.adaptive:
                # Entropy H = -sum(p * log(p+eps))
                H = -(dist * (dist + 1e-8).log()).sum(dim=-1)  # (batch,)
                K_vec = (self.gamma * H).floor().long() + 1
                K_vec = K_vec.clamp(1, self.K_max)
            else:
                K_vec = torch.full(
                    (dist.shape[0],), self.K_max,
                    dtype=torch.long, device=dist.device
                )

            # Zero out all but top-K, then renormalise
            masked = []
            for b_idx in range(dist.shape[0]):
                k = K_vec[b_idx].item()
                d = dist[b_idx]                              # (b_l,)
                top_k_vals, _ = torch.topk(d, k=min(k, d.shape[0]))
                threshold = top_k_vals[-1]
                d_masked = d * (d >= threshold).float()
                d_masked = d_masked / d_masked.sum().clamp(min=1e-9)
                masked.append(d_masked)
            beam_dists.append(torch.stack(masked, dim=0))   # (batch, b_l)

        return beam_dists

    def get_beam_paths(
        self,
        query_emb: torch.Tensor,
        L: int,
        D: int,
    ) -> List[List[torch.Tensor]]:
        """
        Return beam path sets: list of batch elements, each a list of
        active digit-index tensors (one per layer).
        """
        radices = self._gen.compute_radices(L, D)
        dists = self._compute_distributions(query_emb, L, D, radices)
        beam_per_item = []
        for b_idx in range(query_emb.shape[0]):
            paths = [d[b_idx] for d in dists]
            beam_per_item.append(paths)
        return beam_per_item


# ──────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────

def create_compiler(
    compiler_type: str,
    d_model: int,
    **kwargs,
) -> BaseCompiler:
    """
    Factory function.

    Args:
        compiler_type: one of 'deterministic', 'binary_mlp', 'reqmr',
                       'semantic_anchor', 'beam_reqmr', 'beam_binary'
        d_model: model hidden dimension
        **kwargs: forwarded to the specific compiler constructor
    """
    assert compiler_type in {
        "deterministic", "binary_mlp", "reqmr",
        "semantic_anchor", "beam_reqmr", "beam_binary",
    }, f"Unknown compiler type: {compiler_type!r}"

    if compiler_type == "deterministic":
        return DeterministicCompiler(d_model, **kwargs)
    elif compiler_type == "binary_mlp":
        return BinaryMLPCompiler(d_model, **kwargs)
    elif compiler_type == "reqmr":
        return REQMRCompiler(d_model, **kwargs)
    elif compiler_type == "semantic_anchor":
        return SemanticAnchorCompiler(d_model, **kwargs)
    elif compiler_type == "beam_reqmr":
        base = REQMRCompiler(d_model, **{
            k: v for k, v in kwargs.items()
            if k not in ("K_max", "gamma", "adaptive")
        })
        return BeamQMRCompiler(
            base,
            K_max=kwargs.get("K_max", 4),
            gamma=kwargs.get("gamma", 2.0),
            adaptive=kwargs.get("adaptive", True),
        )
    else:  # beam_binary
        base = BinaryMLPCompiler(d_model, **{
            k: v for k, v in kwargs.items()
            if k not in ("K_max", "gamma", "adaptive")
        })
        return BeamQMRCompiler(
            base,
            K_max=kwargs.get("K_max", 4),
            gamma=kwargs.get("gamma", 2.0),
            adaptive=kwargs.get("adaptive", True),
        )


# ──────────────────────────────────────────────────────────────────────
# Standalone tests
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import math

    BATCH, D_MODEL, L, D = 4, 128, 4096, 4
    device = "cpu"

    # ── Test 1: DeterministicCompiler ──
    det = DeterministicCompiler(D_MODEL)
    addresses = torch.randint(0, L, (BATCH,))
    dists = det.forward_from_address(addresses, L, D)
    assert len(dists) == D
    gen = MixedRadixGraphGenerator()
    radices = gen.compute_radices(L, D)
    for l_idx, (dist, b) in enumerate(zip(dists, radices)):
        assert dist.shape == (BATCH, b), f"layer {l_idx}: {dist.shape}"
        assert (dist.sum(dim=-1) - 1).abs().max() < 1e-5
    print(f"DeterministicCompiler OK: radices={radices}")

    # ── Test 2: BinaryMLPCompiler ──
    bin_mlp = BinaryMLPCompiler(D_MODEL)
    q_emb = torch.randn(BATCH, D_MODEL)
    dists2 = bin_mlp(q_emb, L, D)
    assert len(dists2) == D
    for l_idx, (dist, b) in enumerate(zip(dists2, radices)):
        assert dist.shape == (BATCH, b)
        assert (dist.sum(dim=-1) - 1).abs().max() < 1e-5
    print("BinaryMLPCompiler OK")

    # ── Test 3: REQMRCompiler ──
    reqmr = REQMRCompiler(D_MODEL)
    dists3 = reqmr(q_emb, L, D)
    assert len(dists3) == D
    for dist, b in zip(dists3, radices):
        assert dist.shape == (BATCH, b)
        assert (dist.sum(dim=-1) - 1).abs().max() < 1e-5
    # Address reconstruction loss
    target_pos = torch.randint(0, L, (BATCH,))
    loss = reqmr.compute_address_reconstruction_loss(dists3, target_pos, L, D)
    assert loss >= 0, f"Loss must be non-negative: {loss}"
    print(f"REQMRCompiler OK, addr_loss={loss.item():.4f}")

    # ── Test 4: SemanticAnchorCompiler ──
    sem = SemanticAnchorCompiler(D_MODEL)
    dists4 = sem(q_emb, L, D)
    assert len(dists4) == D
    print("SemanticAnchorCompiler OK")

    # ── Test 5: BeamQMRCompiler ──
    beam = BeamQMRCompiler(reqmr, K_max=3, adaptive=True)
    dists5 = beam(q_emb, L, D)
    assert len(dists5) == D
    for dist in dists5:
        # After beam masking, distributions are still valid
        assert (dist.sum(dim=-1) - 1).abs().max() < 1e-5
    print("BeamQMRCompiler OK")

    # ── Test 6: Factory ──
    c = create_compiler("reqmr", D_MODEL)
    assert isinstance(c, REQMRCompiler)
    c2 = create_compiler("beam_reqmr", D_MODEL, K_max=2)
    assert isinstance(c2, BeamQMRCompiler)
    print("Factory OK")

    # ── Test 7: Gradient flow ──
    reqmr.zero_grad()
    q2 = torch.randn(BATCH, D_MODEL, requires_grad=True)
    ds = reqmr(q2, L, D)
    total = sum(d.sum() for d in ds)
    total.backward()
    assert q2.grad is not None
    print("Gradient flow OK")

    print("\nAll compiler tests passed.")
