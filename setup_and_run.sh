#!/bin/bash
set -e  # Exit on error

echo "=== SuccinctBound Experiment Setup and Run ==="
echo ""

# Step 1: Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "[1/5] Creating virtual environment..."
    python3 -m venv venv
    echo "  Virtual environment created."
else
    echo "[1/5] Virtual environment already exists."
fi

# Step 2: Install dependencies
echo ""
echo "[2/5] Installing dependencies..."
./venv/bin/pip install torch torchvision transformers datasets numpy scipy tqdm pyyaml requests matplotlib seaborn pandas jinja2 2>&1 | tail -10
echo "  Dependencies installed."

# Step 3: Test import
echo ""
echo "[3/5] Testing imports..."
./venv/bin/python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>&1
./venv/bin/python3 -c "from models.architectures import VanillaTransformer; print('Models import OK')" 2>&1
./venv/bin/python3 -c "from benchmarks.arithmetic import ArithmeticDataset; print('Benchmarks import OK')" 2>&1

# Step 4: Run a single test experiment (small scale)
echo ""
echo "[4/5] Running test experiment (Vanilla + Arithmetic, 2 epochs)..."
./venv/bin/python3 experiments/train.py \
    --config config.yaml \
    --architecture vanilla \
    --benchmark arithmetic \
    --device cpu 2>&1 | tail -30

# Step 5: If test succeeds, run all experiments
echo ""
echo "[5/5] Test complete. To run all experiments, execute:"
echo "  ./venv/bin/python3 experiments/run_all_experiments.py --sequential --device cpu"
echo ""
echo "Or for GPU (if available):"
echo "  ./venv/bin/python3 experiments/run_all_experiments.py --sequential --device cuda"
echo ""
echo "After experiments complete, generate tables with:"
echo "  ./venv/bin/python3 experiments/collect_results.py"
