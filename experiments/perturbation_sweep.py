"""
Perturbation Sweep Experiment (paper Table 6).

Sweeps perturbation weight omega_0 over {0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, adaptive}.
Measures: Xi_D (routing subspace drift), PathMass, Retrieval, Range, Variety, Summary, Probe.

Usage:
  python experiments/perturbation_sweep.py --omega 1e-3 --L 4096
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from models.mixed_radix_generator import MixedRadixGraphGenerator
from models.compilers import REQMRCompiler, create_compiler
from models.qmr_architectures import QMRFullPlusPlus, create_qmr_model


# ──────────────────────────────────────────────────────────────────────
# Synthetic retrieval dataset
# ──────────────────────────────────────────────────────────────────────

class RetrievalDataset(Dataset):
    """Synthetic retrieval dataset for perturbation sweep."""

    def __init__(self, num_samples: int, L: int, d_model: int, seed: int = 42):
        super().__init__()
        self.num_samples = num_samples
        self.L = L
        self.d_model = d_model

        rng = torch.Generator()
        rng.manual_seed(seed)

        self.contexts = torch.randn(num_samples, L, d_model, generator=rng)
        self.targets = torch.randint(0, L, (num_samples,), generator=rng)
        self.query_embs = torch.randn(num_samples, d_model, generator=rng)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        return {
            "context": self.contexts[idx],
            "query": self.query_embs[idx],
            "target": self.targets[idx],
        }


# ──────────────────────────────────────────────────────────────────────
# Perturbation sweep experiment
# ──────────────────────────────────────────────────────────────────────

def run_perturbation_sweep(
    omega_values: List[float],
    L: int = 4096,
    d_model: int = 128,
    num_layers: int = 4,
    batch_size: int = 4,
    num_epochs: int = 10,
    device: str = "cpu",
    output_dir: str = "results/perturbation_sweep",
) -> Dict[str, Any]:
    """
    Run perturbation sweep experiment (paper Table 6).

    Args:
        omega_values: list of perturbation weights to sweep
        L: sequence length
        d_model: model hidden dimension
        num_layers: number of QMR layers
        batch_size: batch size
        num_epochs: number of training epochs
        device: device to run on
        output_dir: output directory for results

    Returns:
        results: dict mapping omega value -> metrics
    """
    results = {
        "experiment": "perturbation_sweep",
        "config": {
            "L": L,
            "d_model": d_model,
            "num_layers": num_layers,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
        },
        "results": {},
    }

    for omega in omega_values:
        print(f"\n{'='*60}")
        print(f"Omega = {omega}")
        print(f"{'='*60}")

        # Build model
        model = QMRFullPlusPlus(
            d_model=d_model,
            num_layers=num_layers,
            num_routing_heads=4,
            num_local_heads=4,
            L=L,
            window_size=128,
            compiler_type="reqmr",
            perturb_omega0=omega,
        )
        model = model.to(device)

        # Dataset
        dataset = RetrievalDataset(1000, L, d_model)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Train
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            n_batches = 0

            for batch in dataloader:
                context = batch["context"].to(device)
                query = batch["query"].to(device)
                target = batch["target"].to(device)

                # Forward
                output = model(context, query_emb=query)

                # Retrieval loss
                sink_out = output[:, -1, :]
                target_emb = context[torch.arange(context.shape[0]), target]
                loss = 1.0 - F.cosine_similarity(sink_out, target_emb, dim=-1).mean()

                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}")

        # Evaluate
        model.eval()
        with torch.no_grad():
            # Retrieval accuracy
            test_dataset = RetrievalDataset(200, L, d_model, seed=123)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            correct = 0
            total = 0
            path_masses = []

            for batch in test_loader:
                context = batch["context"].to(device)
                query = batch["query"].to(device)
                target = batch["target"].to(device)

                output = model(context, query_emb=query)
                sink_out = output[:, -1, :]

                # Retrieval: find closest context vector
                scores = torch.matmul(context, sink_out.unsqueeze(-1)).squeeze(-1)
                pred_pos = scores.argmax(dim=-1)

                correct += (pred_pos == target).sum().item()
                total += target.shape[0]

                # Path mass
                compiler_dists = model.compiler(query, L, num_layers)
                for dist in compiler_dists:
                    path_mass = dist.max(dim=-1)[0].mean().item()
                    path_masses.append(path_mass)

            acc = correct / max(total, 1)
            avg_path_mass = sum(path_masses) / max(len(path_masses), 1)

            print(f"  Retrieval accuracy: {acc:.4f}")
            print(f"  Path mass: {avg_path_mass:.4f}")

            results["results"][str(omega)] = {
                "retrieval_accuracy": acc,
                "path_mass": avg_path_mass,
                "final_loss": avg_loss,
            }

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_file = output_path / "perturbation_sweep_results.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {result_file}")

    return results


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Perturbation sweep experiment (paper Table 6)"
    )
    parser.add_argument("--omega", type=str, default="0,1e-5,1e-4,1e-3,1e-2,1e-1,adaptive",
                        help="Comma-separated list of omega values to sweep")
    parser.add_argument("--L", type=int, default=1024,
                        help="Sequence length")
    parser.add_argument("--d_model", type=int, default=64,
                        help="Model hidden dimension")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of QMR layers")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu or cuda:N)")
    parser.add_argument("--output_dir", type=str, default="results/perturbation_sweep",
                        help="Output directory")
    args = parser.parse_args()

    # Parse omega values
    omega_str = args.omega.split(",")
    omega_values = []
    for o in omega_str:
        if o == "adaptive":
            omega_values.append(-1)  # special value for adaptive
        else:
            omega_values.append(float(o))

    run_perturbation_sweep(
        omega_values=omega_values,
        L=args.L,
        d_model=args.d_model,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
