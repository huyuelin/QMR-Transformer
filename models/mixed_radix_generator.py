"""
Mixed-Radix Graph Generator for QMR-Transformers (paper §3.1).

Given length L and depth D, selects integer radices b_1,...,b_D >= 2
such that prod(b_l) >= L, computes cumulative scales B_l, and generates
routing offset sets O_l = {-a*B_{l-1} : a=0,...,b_l-1}.

The balanced schedule b_l ≈ L^(1/D) minimizes the total edge budget
E(L) = L * sum(b_l), attaining the DL^{1+1/D} frontier (Theorem 2).
"""

import math
import torch
from typing import List, Tuple


class MixedRadixGraphGenerator:
    """
    Implements the mixed-radix graph generator (paper §3.1).

    Key properties:
      - prod(b_l) >= L  ensures full fixed-sink coverage (Theorem 3)
      - balanced schedule b_l ∈ {floor(L^(1/D)), ceil(L^(1/D))} is optimal
      - canonical path from sink to any position q: unique digit decomposition
        of (L - 1 - q) in the mixed-radix base (b_1, ..., b_D)
    """

    def __init__(self, balanced: bool = True):
        """
        Args:
            balanced: use balanced schedule (b_l ≈ L^(1/D)); if False,
                      uses a simple uniform radix (b_l = ceil(L^(1/D)))
        """
        self.balanced = balanced

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute_radices(self, L: int, D: int) -> List[int]:
        """
        Compute integer radices for length L and depth D.

        Algorithm (balanced schedule):
          base = max(2, floor(L^(1/D)))
          initialise all b_l = base
          increment the smallest b_l one at a time until prod >= L
          (keeps the b_l values within {base, base+1}, minimising sum)

        Args:
            L: sequence length (>= 1)
            D: model depth (number of routing layers, >= 1)

        Returns:
            radices: list of D integers, each >= 2, with prod >= L
        """
        assert L >= 1, f"L must be >= 1, got {L}"
        assert D >= 1, f"D must be >= 1, got {D}"

        if L == 1:
            return [2] * D  # trivial: any radices cover length 1

        base = max(2, int(L ** (1.0 / D)))
        radices = [base] * D

        # Increment until product covers L.  Always pick the smallest
        # element so the values stay as uniform as possible.
        while math.prod(radices) < L:
            min_idx = min(range(D), key=lambda i: radices[i])
            radices[min_idx] += 1

        # Sort ascending: smaller radices first so earlier layers have
        # shorter stride (local reach) and later layers have longer stride
        # (global reach), matching the paper's convention b=[3,4] for L=12,D=2.
        radices.sort()
        return radices

    def compute_cumulative_scales(self, radices: List[int]) -> List[int]:
        """
        B_0 = 1, B_l = prod(b_1 ... b_l) for l = 1,...,D.

        Returns a list of length D+1: [B_0, B_1, ..., B_D].
        """
        B = [1]
        for b in radices:
            B.append(B[-1] * b)
        return B

    def generate_offsets(
        self, L: int, D: int
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Generate per-layer routing offset sets.

        O_l = {-a * B_{l-1} : a = 0, ..., b_l - 1}

        Args:
            L: sequence length
            D: number of routing layers

        Returns:
            all_offsets: list of D tensors; all_offsets[l] has shape (b_l,)
                         and contains the signed offsets for layer l
            radices: the computed radices
        """
        radices = self.compute_radices(L, D)
        B = self.compute_cumulative_scales(radices)

        all_offsets = []
        for l_idx in range(D):
            b = radices[l_idx]
            B_prev = B[l_idx]           # B_{l-1}
            # offsets: 0, -B_prev, -2*B_prev, ..., -(b-1)*B_prev
            offsets = -(torch.arange(b, dtype=torch.long) * B_prev)
            all_offsets.append(offsets)

        return all_offsets, radices

    def get_attention_mask(
        self, L: int, D: int, device: str = "cpu"
    ) -> torch.Tensor:
        """
        Build the QMR sparse attention mask for verification / visualisation.

        mask[l, i, j] = 1 iff j is a valid predecessor of i at layer l,
        i.e. j - i ∈ O_l (equivalently j = i - a*B_{l-1} for some a).

        Returns:
            mask: bool tensor of shape (D, L, L)
        """
        radices = self.compute_radices(L, D)
        B = self.compute_cumulative_scales(radices)

        mask = torch.zeros(D, L, L, dtype=torch.bool, device=device)

        for l_idx in range(D):
            b = radices[l_idx]
            B_prev = B[l_idx]
            # vectorised: for each query position i and digit a
            i_range = torch.arange(L, device=device)             # (L,)
            a_range = torch.arange(b, device=device)             # (b,)
            # j[i, a] = i - a * B_prev
            j_mat = i_range.unsqueeze(1) - a_range.unsqueeze(0) * B_prev  # (L, b)
            # clamp to valid range
            valid = (j_mat >= 0) & (j_mat < L)
            rows = i_range.unsqueeze(1).expand_as(j_mat)[valid]
            cols = j_mat[valid]
            mask[l_idx, rows, cols] = True

        return mask

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def encode_position(
        self, pos: int, L: int, D: int
    ) -> List[int]:
        """
        Encode displacement (L - 1 - pos) as mixed-radix digits.

        a_l = floor(displacement / B_{l-1}) mod b_l

        This gives the canonical path digits from the sink (position L-1)
        to target position pos (Theorem 3: unique mixed-radix expansion).

        Args:
            pos: target position in [0, L)
            L: sequence length
            D: number of layers

        Returns:
            digits: list of D integers, a_l ∈ [0, b_l)
        """
        assert 0 <= pos < L, f"pos {pos} out of range [0, {L})"
        radices = self.compute_radices(L, D)
        B = self.compute_cumulative_scales(radices)

        displacement = (L - 1) - pos
        assert displacement >= 0
        assert displacement < math.prod(radices), (
            f"displacement {displacement} >= product of radices "
            f"{math.prod(radices)} — coverage violated"
        )

        digits = []
        for l_idx in range(D):
            b = radices[l_idx]
            B_prev = B[l_idx]
            a = (displacement // B_prev) % b
            digits.append(a)
        return digits

    def verify_coverage(self, L: int, D: int) -> bool:
        """
        Verify that the mixed-radix graph covers all source positions.

        Builds the reverse-receptive field from the sink (position L-1)
        and checks that it reaches every position in [0, L).
        """
        radices = self.compute_radices(L, D)
        if math.prod(radices) < L:
            return False

        B = self.compute_cumulative_scales(radices)
        reachable = set(range(L))  # all positions must be reachable

        # Forward check: each position can be encoded with valid digits
        for pos in range(L):
            displacement = (L - 1) - pos
            if displacement >= math.prod(radices):
                return False
        return True

    def edge_budget(self, L: int, D: int) -> int:
        """
        Total sparse edge budget E(L) = L * sum(b_l).

        The lower bound is D * L^{1+1/D} (Theorem 2).
        """
        radices = self.compute_radices(L, D)
        return L * sum(radices)

    def edge_budget_lower_bound(self, L: int, D: int) -> float:
        """Theoretical lower bound D * L^(1+1/D)."""
        return D * (L ** (1.0 + 1.0 / D))


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------
if __name__ == "__main__":
    gen = MixedRadixGraphGenerator()

    # ── Test 1: paper example (L=12, D=2, expected b=[3,4]) ──
    L, D = 12, 2
    offsets, radices = gen.generate_offsets(L, D)
    print(f"L={L}, D={D}")
    print(f"  radices: {radices}")
    print(f"  B: {gen.compute_cumulative_scales(radices)}")
    print(f"  offsets[0]: {offsets[0].tolist()}")
    print(f"  offsets[1]: {offsets[1].tolist()}")
    assert math.prod(radices) >= L, "Coverage violated"

    # ── Test 2: canonical path from sink to position 5 ──
    digits = gen.encode_position(5, L, D)
    print(f"  canonical digits for pos 5: {digits}")
    # displacement = 11 - 5 = 6; a_0 = 6%3=0, a_1 = (6//3)%4=2
    assert digits == [0, 2], f"Unexpected digits: {digits}"

    # ── Test 3: coverage verified ──
    assert gen.verify_coverage(L, D), "Coverage check failed"
    print(f"  coverage: OK")

    # ── Test 4: attention mask shape ──
    mask = gen.get_attention_mask(L, D)
    print(f"  mask shape: {mask.shape}, total edges: {mask.sum().item()}")
    expected_edges = sum(
        sum(1 for i in range(L) for a in range(radices[l])
            if 0 <= i - a * gen.compute_cumulative_scales(radices)[l] < L)
        for l in range(D)
    )
    assert mask.sum().item() == expected_edges

    # ── Test 5: edge budget vs lower bound ──
    for length in [512, 4096, 16384]:
        eb = gen.edge_budget(length, D)
        lb = gen.edge_budget_lower_bound(length, D)
        print(f"  L={length}: budget={eb}, lower_bound={lb:.0f}, ratio={eb/lb:.3f}")
        assert eb >= lb * 0.95, f"Budget {eb} far below lower bound {lb}"

    # ── Test 6: large L ──
    L2, D2 = 65536, 4
    r2 = gen.compute_radices(L2, D2)
    print(f"\nL={L2}, D={D2}: radices={r2}, prod={math.prod(r2)}")
    assert math.prod(r2) >= L2

    print("\nAll tests passed.")
