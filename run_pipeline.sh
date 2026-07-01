#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# SuccinctBound Full Reproduction Pipeline
# Runs on 4090_hyl server (4 × RTX 4090)
# ═══════════════════════════════════════════════════════════════════════

set -e

WORK_DIR="$HOME/succinct_bound"
RESULTS_DIR="$WORK_DIR/results"
TABLES_DIR="$WORK_DIR/tables"

echo "════════════════════════════════════════════════════════════════"
echo "  SuccinctBound: Full Paper Reproduction"
echo "  Target: 6 architectures × 3 benchmarks = 18 experiments"
echo "  GPUs: 4 × RTX 4090"
echo "════════════════════════════════════════════════════════════════"

mkdir -p "$RESULTS_DIR" "$TABLES_DIR"

# ─── Phase 1: Run experiments in parallel across GPUs ───
echo ""
echo "Phase 1: Running 18 experiments across 4 GPUs..."
echo ""

# Distribute: 6 experiments per GPU (approx), 3 benchmarks × 2 archs each
# GPU 0: vanilla + sparse (all 3 benchmarks)
# GPU 1: relative + nope (all 3 benchmarks)
# GPU 2: ssm + rnn (all 3 benchmarks)
# GPU 3: reserved for Lean verification / overflow

run_on_gpu() {
    local gpu=$1
    local arch=$2
    local bench=$3
    echo "  [GPU:$gpu] Starting $arch × $bench"
    python3 "$WORK_DIR/run_full_experiments.py" \
        --device "cuda:$gpu" \
        --arch "$arch" \
        --bench "$bench" \
        --epochs 50 \
        --batch-size 128 \
        --lr 3e-4 \
        --seed 42 \
        > "$RESULTS_DIR/${arch}_${bench}.log" 2>&1
    echo "  [GPU:$gpu] Done: $arch × $bench"
}

# Launch in parallel batches
echo "Batch 1/3: GPU 0-3 running vanilla/sparse/relative/nope on dyck..."
run_on_gpu 0 vanilla dyck &
run_on_gpu 1 sparse dyck &
run_on_gpu 2 relative dyck &
run_on_gpu 3 nope dyck &
wait

echo "Batch 2/3: GPU 0-3 running ssm/rnn + arithmetic..."
run_on_gpu 0 ssm dyck &
run_on_gpu 1 rnn dyck &
run_on_gpu 2 vanilla arithmetic &
run_on_gpu 3 sparse arithmetic &
wait

echo "Batch 3/3: remaining experiments..."
run_on_gpu 0 relative arithmetic &
run_on_gpu 1 nope arithmetic &
run_on_gpu 2 ssm arithmetic &
run_on_gpu 3 rnn arithmetic &
wait

echo "Batch 4/5: counting benchmark..."
run_on_gpu 0 vanilla counting &
run_on_gpu 1 sparse counting &
run_on_gpu 2 relative counting &
run_on_gpu 3 nope counting &
wait

echo "Batch 5/5: remaining counting..."
run_on_gpu 0 ssm counting &
run_on_gpu 1 rnn counting &
wait

# ─── Phase 2: Aggregate results ───
echo ""
echo "Phase 2: Aggregating results and generating tables..."
python3 -c "
import json, os
from pathlib import Path

results_dir = Path('$RESULTS_DIR')
all_results = []
for f in sorted(results_dir.glob('*_results.json')):
    with open(f) as fp:
        all_results.append(json.load(fp))

with open(results_dir / 'all_results.json', 'w') as fp:
    json.dump(all_results, fp, indent=2)

print(f'Aggregated {len(all_results)} experiment results.')
"

# Generate tables
cd "$WORK_DIR"
python3 run_full_experiments.py --device cpu --epochs 0 2>/dev/null || \
python3 -c "
import sys
sys.path.insert(0, '.')
from run_full_experiments import generate_latex_tables, ARCHITECTURES
import json
with open('results/all_results.json') as f:
    results = json.load(f)
generate_latex_tables(results, 'tables')
"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  DONE! Results in: $RESULTS_DIR"
echo "  Tables in: $TABLES_DIR"
echo "════════════════════════════════════════════════════════════════"
