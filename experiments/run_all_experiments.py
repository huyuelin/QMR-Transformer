"""
Batch experiment runner for Succinctness-Generalization Plane.

Runs all combinations of architecture variants and benchmarks,
then collects and summarizes results.
"""

import subprocess
import argparse
import yaml
from pathlib import Path
import sys

ARCHITECTURES = ['vanilla', 'sparse', 'relative', 'nope', 'ssm', 'rnn']
BENCHMARKS = ['arithmetic', 'dyck', 'counting']


def run_experiment(architecture: str, benchmark: str, config_path: str, device: str = "cuda"):
    """Run a single experiment."""
    cmd = [
        sys.executable,
        'train.py',
        '--config', config_path,
        '--architecture', architecture,
        '--benchmark', benchmark,
        '--device', device,
    ]
    
    print(f"\n{'='*80}")
    print(f"Running: architecture={architecture}, benchmark={benchmark}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Run all SuccinctBound experiments')
    parser.add_argument('--config', type=str, default='../config.yaml', help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--architectures', type=str, nargs='+', default=ARCHITECTURES,
                       help='Architecture variants to run')
    parser.add_argument('--benchmarks', type=str, nargs='+', default=BENCHMARKS,
                       help='Benchmarks to run')
    parser.add_argument('--sequential', action='store_true', help='Run experiments sequentially')
    args = parser.parse_args()
    
    # Load config to get results directory
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    results_dir = Path(config.get('results_dir', './results'))
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create experiment plan
    experiments = []
    for arch in args.architectures:
        for bench in args.benchmarks:
            experiments.append((arch, bench))
    
    print(f"\nExperiment Plan:")
    print(f"  Architectures: {args.architectures}")
    print(f"  Benchmarks: {args.benchmarks}")
    print(f"  Total experiments: {len(experiments)}")
    print(f"  Results will be saved to: {results_dir}")
    print()
    
    # Run experiments
    if args.sequential or len(experiments) == 1:
        # Sequential execution
        results = {}
        for arch, bench in experiments:
            returncode = run_experiment(arch, bench, args.config, args.device)
            results[(arch, bench)] = 'success' if returncode == 0 else 'failed'
        
        # Print summary
        print(f"\n{'='*80}")
        print("Experiment Summary:")
        print(f"{'='*80}")
        for (arch, bench), status in results.items():
            print(f"  {arch:12s} + {bench:12s}: {status}")
        print()
    else:
        print("\nFor parallel execution, use:")
        print("  python run_all_experiments.py --sequential")
        print("\nOr manually run experiments in parallel with:")
        for arch, bench in experiments:
            print(f"  python experiments/train.py --architecture {arch} --benchmark {bench} &")
        print()
        
        # Still run sequentially as fallback
        returncode = run_experiment(args.architectures[0], args.benchmarks[0], args.config, args.device)


if __name__ == '__main__':
    main()
