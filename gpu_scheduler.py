#!/usr/bin/env python3
"""
GPU Scheduler for 4090_hyl server (124.220.80.30:7040).

Runs QMR experiments in parallel across available GPUs.
Updated for QMR-Transformer reproduction (paper aaai2026).

Server config:
  Host: 124.220.80.30
  Port: 7040
  User: hyl
  Remote dir: ~/qmr_experiments
"""

import subprocess
import time
import argparse
import json
from pathlib import Path
import sys
from typing import List, Dict, Optional
import os

# 4090_hyl server config
SERVER_CONFIG = {
    'hostname': '124.220.80.30',
    'port': 7040,
    'username': 'hyl',
    'remote_dir': '~/qmr_experiments',
    'gpu_count': 4,           # adjust based on actual server
    'gpu_memory_mib': 24000,  # approximate 24GB 4090
}


def check_server_status():
    """Check server and GPU status."""
    print("Checking server status...")
    
    cmd = f"ssh -p {SERVER_CONFIG['port']} {SERVER_CONFIG['username']}@{SERVER_CONFIG['hostname']} 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader'"
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Server connection successful")
        print("\nGPU Status:")
        print(result.stdout)
        return True
    else:
        print(f"✗ Server connection failed: {result.stderr}")
        return False


def sync_code_to_server():
    """Sync code to SenseTime server."""
    print("\nSyncing code to server...")
    
    remote = f"{SERVER_CONFIG['username']}@{SERVER_CONFIG['hostname']}:{SERVER_CONFIG['remote_dir']}"
    cmd = f"rsync -avz -e 'ssh -p {SERVER_CONFIG['port']}' --exclude 'venv' --exclude '__pycache__' ./ {remote}/"
    
    print(f"Running: {cmd[:80]}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Code synced successfully")
        return True
    else:
        print(f"✗ Sync failed: {result.stderr}")
        return False


def run_experiment_on_gpu(architecture, benchmark, gpu_id):
    """Run a single experiment on a specific GPU."""
    
    cmd = f"""ssh -p {SERVER_CONFIG['port']} {SERVER_CONFIG['username']}@{SERVER_CONFIG['hostname']} '
cd {SERVER_CONFIG['remote_dir']} &&
source venv/bin/activate &&
CUDA_VISIBLE_DEVICES={gpu_id} python train_full.py \
    --architecture {architecture} \
    --benchmark {benchmark} \
    --device cuda:0 \
    --epochs 10 \
    --batch-size 64
'"""
    
    print(f"  [GPU {gpu_id}] Starting: {architecture} + {benchmark}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"  [GPU {gpu_id}] ✓ Completed: {architecture} + {benchmark}")
        return True, result.stdout
    else:
        print(f"  [GPU {gpu_id}] ✗ Failed: {architecture} + {benchmark}")
        print(f"  Error: {result.stderr[:200]}")
        return False, result.stderr


def run_all_experiments_parallel():
    """Run all experiments in parallel across 4 GPUs."""
    
    architectures = ['vanilla', 'sparse', 'relative', 'nope', 'ssm', 'rnn']
    benchmarks = ['arithmetic', 'dyck', 'counting']
    
    experiments = []
    for arch in architectures:
        for bench in benchmarks:
            experiments.append((arch, bench))
    
    print(f"\n{'='*60}")
    print(f"Running {len(experiments)} experiments on 4x RTX 4090 GPUs")
    print(f"{'='*60}\n")
    
    # Check server status
    if not check_server_status():
        print("\nPlease check server connection and try again.")
        return False
    
    # Sync code
    if not sync_code_to_server():
        print("\nFailed to sync code. Please check and try again.")
        return False
    
    # Run experiments in parallel (4 at a time)
    results = []
    running = {}  # gpu_id -> (arch, bench, process)
    
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def run_on_gpu(gpu_id, arch, bench):
        """Run experiment on a specific GPU."""
        success, output = run_experiment_on_gpu(arch, bench, gpu_id)
        return gpu_id, arch, bench, success, output
    
    # Submit experiments
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        
        for i, (arch, bench) in enumerate(experiments):
            gpu_id = i % 4  # Round-robin GPU assignment
            future = executor.submit(run_on_gpu, gpu_id, arch, bench)
            futures.append(future)
            time.sleep(1)  # Stagger submissions
        
        # Wait for all to complete
        for future in as_completed(futures):
            gpu_id, arch, bench, success, output = future.result()
            results.append({
                'architecture': arch,
                'benchmark': bench,
                'gpu_id': gpu_id,
                'success': success
            })
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Experiment Summary")
    print(f"{'='*60}")
    
    successful = sum(1 for r in results if r['success'])
    print(f"  Total: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {len(results) - successful}")
    
    for result in results:
        status = "✓" if result['success'] else "✗"
        print(f"  {status} {result['architecture']:12s} + {result['benchmark']:12s} (GPU {result['gpu_id']})")
    
    return successful == len(results)


def sync_results_back():
    """Sync results back from server."""
    print("\nSyncing results from server...")
    
    remote = f"{SERVER_CONFIG['username']}@{SERVER_CONFIG['hostname']}:{SERVER_CONFIG['remote_dir']}/results"
    local = "./results"
    
    cmd = f"rsync -avz -e 'ssh -p {SERVER_CONFIG['port']}' {remote}/ {local}/"
    
    print(f"Running: {cmd[:80]}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Results synced successfully")
        return True
    else:
        print(f"✗ Sync failed: {result.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description='GPU Scheduler for SuccinctBound')
    parser.add_argument('--check', action='store_true', help='Check server status')
    parser.add_argument('--sync', action='store_true', help='Sync code to server')
    parser.add_argument('--run', action='store_true', help='Run all experiments')
    parser.add_argument('--sync-results', action='store_true', help='Sync results back')
    parser.add_argument('--all', action='store_true', help='Do all steps')
    
    args = parser.parse_args()
    
    if args.all:
        args.check = args.sync = args.run = args.sync_results = True
    
    if args.check:
        check_server_status()
    
    if args.sync:
        sync_code_to_server()
    
    if args.run:
        run_all_experiments_parallel()
    
    if args.sync_results:
        sync_results_back()
    
    if not any([args.check, args.sync, args.run, args.sync_results, args.all]):
        print("No action specified. Use --check, --sync, --run, --sync-results, or --all")
        print("\nExample usage:")
        print("  python gpu_scheduler.py --all  # Do everything")
        print("  python gpu_scheduler.py --check --sync  # Check and sync only")


if __name__ == '__main__':
    main()
