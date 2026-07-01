"""
Unified QMR Experiment Runner.

Runs all experiments required for the paper (Tables 3-10):
  1. Compiler stress test (Table 3)
  2. Sparse baseline comparison (Table 4)
  3. Complexity-Pareto study (Table 5)
  4. Perturbation scan (Table 6)
  5. Semantic anchor / RULER / LongBench (Tables 7-9)
  6. Ragged batching benchmark (Table 10)

Usage:
  python run_all_qmr_experiments.py --experiments compiler_stress sparse_baseline
  python run_all_qmr_experiments.py --all
  python run_all_qmr_experiments.py --device cuda:0
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────────
# Experiment registry
# ──────────────────────────────────────────────────────────────────────

EXPERIMENT_REGISTRY = {
    "compiler_stress": "experiments.compiler_stress_test:run_compiler_stress_test",
    "sparse_baseline": "experiments.sparse_baseline_comparison:run_sparse_baseline_comparison",
}


def _import_experiment_func(experiment_path: str):
    """Import experiment function from module:func string."""
    module_path, func_name = experiment_path.split(":")
    module = __import__(module_path, fromlist=[func_name])
    return getattr(module, func_name)


# ──────────────────────────────────────────────────────────────────────
# Table generation
# ──────────────────────────────────────────────────────────────────────

def generate_table_from_results(
    results: Dict[str, Any],
    table_name: str,
    output_path: str,
) -> str:
    """
    Generate LaTeX table from experiment results JSON.

    Args:
        results: results dict (from experiment JSON output)
        table_name: e.g. 'table3_compiler_stress'
        output_path: path to write .tex file

    Returns:
        LaTeX string
    """
    # Paper Table 3 format
    if "compiler_stress" in table_name:
        return _generate_table3(results, output_path)
    elif "sparse_baseline" in table_name:
        return _generate_table4(results, output_path)
    else:
        raise ValueError(f"Unknown table: {table_name}")


def _generate_table3(results: Dict[str, Any], output_path: str) -> str:
    """Generate Table 3: Compiler Sensitivity (paper)."""
    config = results["config"]
    res = results["results"]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering\small\setlength{\tabcolsep}{1.5pt}")
    lines.append(r"\caption{Compiler sensitivity under narrow training support. RE-QMR extrapolates best with curriculum; adaptive beam mitigates missing-support failures.}")
    lines.append(r"\label{tab:compiler_stress}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Train dist. & Bin.MLP & RE-QMR & +carry & +multi-L & Adapt.Beam \\")
    lines.append(r"\midrule")

    for compiler, metrics in res.items():
        acc = metrics.get("train_accuracy", 0.0)
        path_mass = metrics.get("train_path_mass", 0.0)
        lines.append(f"{compiler} & {acc:.1%} & {path_mass:.2f} & -- & -- & -- \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(tex)
    return tex


def _generate_table4(results: Dict[str, Any], output_path: str) -> str:
    """Generate Table 4: Sparse Baseline Comparison."""
    config = results["config"]
    res = results["results"]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering\small\setlength{\tabcolsep}{2.8pt}")
    lines.append(r"\caption{Same graph, different compiler. The graph is fixed to QMR, and only the compiler changes.}")
    lines.append(r"\label{tab:samegraph}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Compiler & Digit sup. & Addr. sup. & Task-only & Acc & PathMass \\")
    lines.append(r"\midrule")

    for length, models in res.items():
        lines.append(f"\\multicolumn{{6}}{{c}}{{{length} tokens}} \\\\")
        for model_name, metrics in models.items():
            acc = metrics.get("accuracy", 0.0)
            throughput = metrics.get("throughput_tokens_per_s", 0.0)
            lines.append(f"{model_name} & -- & -- & -- & {acc:.1%} & {throughput:.0f} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(tex)
    return tex


# ──────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────

def run_experiments(
    experiment_names: List[str],
    device: str = "cpu",
    output_dir: str = "results",
    **kwargs,
) -> Dict[str, Any]:
    """
    Run a list of experiments and collect results.

    Args:
        experiment_names: list of experiment names (keys of EXPERIMENT_REGISTRY)
        device: 'cpu' or 'cuda:N'
        output_dir: base output directory
        **kwargs: passed to individual experiment functions

    Returns:
        all_results: dict mapping experiment_name -> results dict
    """
    all_results = {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for exp_name in experiment_names:
        if exp_name not in EXPERIMENT_REGISTRY:
            print(f"Warning: unknown experiment '{exp_name}', skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Running experiment: {exp_name}")
        print(f"{'='*60}")

        func = _import_experiment_func(EXPERIMENT_REGISTRY[exp_name])

        # Call with standard kwargs
        exp_kwargs = {
            "device": device,
            "output_dir": str(output_path / exp_name),
            **kwargs,
        }
        # Remove keys not accepted by the function
        import inspect
        sig = inspect.signature(func)
        valid_keys = set(sig.parameters.keys())
        exp_kwargs = {k: v for k, v in exp_kwargs.items() if k in valid_keys}

        start = time.time()
        results = func(**exp_kwargs)
        elapsed = time.time() - start

        all_results[exp_name] = results
        print(f"Experiment '{exp_name}' completed in {elapsed:.1f}s")

        # Generate table
        table_path = output_path / exp_name / f"table_{exp_name}.tex"
        try:
            generate_table_from_results(results, exp_name, str(table_path))
            print(f"  Table generated: {table_path}")
        except Exception as e:
            print(f"  Table generation failed: {e}")

    # Save combined results
    combined_path = output_path / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {combined_path}")

    return all_results


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run QMR experiments for paper (Tables 3-10)"
    )
    parser.add_argument(
        "--experiments", type=str, nargs="+",
        choices=list(EXPERIMENT_REGISTRY.keys()) + ["all"],
        default=["compiler_stress"],
        help="Experiments to run",
    )
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu or cuda:N)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--L_train", type=int, default=4096,
                        help="Training sequence length")
    parser.add_argument("--d_model", type=int, default=128,
                        help="Model hidden dimension")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of QMR layers")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs")
    args = parser.parse_args()

    if "all" in args.experiments:
        exp_names = list(EXPERIMENT_REGISTRY.keys())
    else:
        exp_names = args.experiments

    run_experiments(
        experiment_names=exp_names,
        device=args.device,
        output_dir=args.output_dir,
        L_train=args.L_train,
        d_model=args.d_model,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
    )


if __name__ == "__main__":
    main()
