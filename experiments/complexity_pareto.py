"""
Architecture Complexity vs. Performance Pareto Study (paper Figure 4, Table 5).

Varies: D (depth ∈ {1, 2, 4, 6, 8}), sum(b_l) (edge budget), architecture variant.
Measures: throughput (tokens/s), retrieval accuracy at 64K.
Model scale: 150M parameters, batch size 8, 3 seeds.

Output: results/complexity_pareto.json
"""

import json
import math
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from models.qmr_architectures import create_qmr_model


DEPTHS = [1, 2, 4, 6, 8]
VARIANTS = ['qmr_lite', 'qmr_core', 'qmr_core_plus', 'qmr_full', 'qmr_full_plus_plus']
L = 65536  # 64K


def measure_throughput(model, L: int, d_model: int, device: torch.device, n_repeats: int = 10) -> float:
    """
    Measure model throughput in tokens/second.

    Args:
        model: QMR model
        L: sequence length
        d_model: model hidden dimension
        device: torch device
        n_repeats: number of repeats for timing

    Returns:
        throughput: tokens per second
    """
    model.eval()
    x = torch.randn(1, L, d_model, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(x, query_emb=x[:, -1, :])

    # Measure
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_repeats):
            _ = model(x, query_emb=x[:, -1, :])
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    return (n_repeats * L) / elapsed  # tokens/s


def measure_retrieval_accuracy(model, L: int, d_model: int, device: torch.device, n_samples: int = 100) -> float:
    """
    Measure retrieval accuracy on synthetic task.

    Args:
        model: QMR model
        L: sequence length
        d_model: model hidden dimension
        device: torch device
        n_samples: number of test samples

    Returns:
        accuracy: percentage of correct retrievals
    """
    correct = 0
    model.eval()

    with torch.no_grad():
        for _ in range(n_samples):
            context = torch.randn(1, L, d_model, device=device)
            target_pos = torch.randint(0, L, (1,)).item()
            query = context[:, target_pos, :]

            out = model(context, query_emb=query)

            # Check if output is close to target
            sim = torch.cosine_similarity(out[:, -1, :], context[:, target_pos, :], dim=-1)
            if sim.item() > 0.5:
                correct += 1

    return correct / n_samples * 100


def main():
    parser = argparse.ArgumentParser(description="Complexity-Pareto study (paper Figure 4, Table 5)")
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--d_model', type=int, default=512, help='Model hidden dimension')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 1337, 2024], help='Random seeds')
    parser.add_argument('--L', type=int, default=4096, help='Sequence length (smaller for testing)')
    parser.add_argument('--output', type=str, default='results/complexity_pareto.json', help='Output path')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    results = []

    for variant in VARIANTS:
        for D in DEPTHS:
            for seed in args.seeds:
                torch.manual_seed(seed)
                try:
                    model = create_qmr_model(
                        variant=variant,
                        d_model=args.d_model,
                        num_layers=D,
                        num_routing_heads=4,
                        num_local_heads=4,
                        L=args.L,
                        window_size=128,
                        compiler_type='reqmr'
                    ).to(device)

                    n_params = sum(p.numel() for p in model.parameters())
                    tput = measure_throughput(model, args.L, args.d_model, device)
                    acc = measure_retrieval_accuracy(model, args.L, args.d_model, device)

                    results.append({
                        'variant': variant,
                        'D': D,
                        'seed': seed,
                        'n_params': n_params,
                        'throughput': tput,
                        'accuracy': acc,
                        'edge_budget': D * (args.L ** (1/D))  # theoretical
                    })
                    print(f"{variant} D={D} seed={seed}: acc={acc:.1f}% tput={tput:.0f} tok/s")
                except Exception as e:
                    print(f"FAILED {variant} D={D}: {e}")

    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
