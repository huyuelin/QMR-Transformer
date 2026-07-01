# QMR-Transformer

**Query-Addressable Mixed-Radix Transformers: Verified Bounds and Efficient Long-Context Retrieval Substrate**

*Yuelin Hu, Zhenbo Yu, Zhengxue Cheng, Wei Liu, Li Song*

Shanghai Jiao Tong University & Shanghai Maritime University

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        QMR-Transformer Pipeline                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input Sequence x₁, x₂, ..., x_L                                            │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              Mixed-Radix Graph Generator                             │    │
│  │  Radices: b₁, b₂, ..., b_D   (∏bₗ ≥ L)                             │    │
│  │  Offsets: O_ℓ = {-a·B_{ℓ-1} : a = 0,...,bₗ-1}                      │    │
│  └────────────────────────────┬────────────────────────────────────────┘    │
│                               │                                              │
│         ┌─────────────────────┼─────────────────────┐                       │
│         │                     │                     │                        │
│         ▼                     ▼                     ▼                        │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐             │
│  │  Layer 1    │    │    Layer 2      │    │    Layer D      │             │
│  │             │    │                 │    │                 │             │
│  │ ┌─────────┐│    │ ┌─────────────┐ │    │ ┌─────────────┐ │             │
│  │ │ Routing ││    │ │   Routing   │ │    │ │   Routing   │ │             │
│  │ │  Heads  ││    │ │    Heads    │ │    │ │    Heads    │ │             │
│  │ │(stride 1)│    │ │ (stride B₁) │ │    │ │(stride B_D-1)│ │             │
│  │ └────┬────┘│    │ └──────┬──────┘ │    │ └──────┬──────┘ │             │
│  │      │     │    │        │        │    │        │        │             │
│  │ ┌────┴────┐│    │ ┌──────┴──────┐ │    │ ┌──────┴──────┐ │             │
│  │ │ Content ││    │ │   Content   │ │    │ │   Content   │ │             │
│  │ │  Heads  ││    │ │    Heads    │ │    │ │    Heads    │ │             │
│  │ │ (local) ││    │ │   (local)   │ │    │ │   (local)   │ │             │
│  │ └────┬────┘│    │ └──────┬──────┘ │    │ └──────┬──────┘ │             │
│  │      │     │    │        │        │    │        │        │             │
│  │   g^r,g^c │    │     g^r,g^c     │    │     g^r,g^c     │             │
│  │   (gates) │    │     (gates)     │    │     (gates)     │             │
│  └─────┬─────┘    └────────┬────────┘    └────────┬────────┘             │
│        └────────────────────┼─────────────────────-┘                       │
│                             ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Query Compiler Family                              │    │
│  │                                                                      │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐  ┌───────┐  │    │
│  │  │Determin- │  │  Binary  │  │  RE-QMR  │  │Semantic │  │ Beam  │  │    │
│  │  │  istic   │  │   MLP    │  │(Norm Addr)│  │ Anchor  │  │  QMR  │  │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └─────────┘  └───────┘  │    │
│  │       q → digits via        q → t̂ ∈ [0,1]   q → block    q → top-k │    │
│  │       index conversion      → â = radix_dec   anchor      paths     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                             │                                                │
│                             ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              Index-Free Sparse Kernel (Triton)                        │    │
│  │  • No materialized index tensor                                      │    │
│  │  • Ragged prefill & decode support                                   │    │
│  │  • O(D·L^{1+1/D}) sparse edge budget (optimal)                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                             │                                                │
│                             ▼                                                │
│                      Output / Retrieval                                       │
└─────────────────────────────────────────────────────────────────────────────┘

Formal Verification Layer (Lean 4 + Seed-Prover):
  ✓ Mixed-radix uniqueness    ✓ Product coverage       ✓ Boundary safety
  ✓ Integer scheduler         ✓ Leakage inequalities   ✓ Beam containment
  ✓ Perturbation recurrence   ✓ Multi-sink coverage
```

## Key Contributions

1. **Edge-Depth Optimality**: We prove that full fixed-sink coverage requires a total sparse edge budget of at least $DL^{1+1/D}$. Mixed-radix masks attain this frontier.

2. **Reachability-Routability Separation**: Full graph coverage does not imply useful routing. Query-conditioned routing is necessary.

3. **Compiler-Conditioned Routing**: A family of query compilers (deterministic, RE-QMR, semantic anchor, beam) that convert graph structure into robust long-context retrieval.

4. **Lean 4 Formal Verification**: Finite combinatorial and algebraic claims verified by the Lean 4 kernel.

5. **Index-Free Sparse Kernel**: Efficient Triton kernel for ragged prefill and decode without materialized index tensors.

## Project Structure

```
QMR-Transformer/
├── models/
│   ├── qmr_architectures.py       # QMR architecture family (Lite → Full++)
│   ├── qmr_transformer_block.py   # QMR block with routing + content heads
│   ├── compilers.py               # Query compiler family
│   └── mixed_radix_generator.py   # Mixed-radix graph construction
├── kernels/
│   └── index_free_kernel.py       # Triton index-free sparse attention kernel
├── benchmarks/                    # Evaluation benchmarks
├── experiments/                   # Training and evaluation scripts
├── lean_proofs/
│   ├── main_theorems.lean         # Core theorems (coverage, separation, leakage)
│   ├── SuccinctBound_mathlib.lean # Mathlib-dependent formalizations
│   └── seed_prover_client.py      # Seed-Prover integration
├── results/                       # Experimental results
├── tables/                        # Generated LaTeX tables
├── config.yaml                    # Experiment configuration
├── train_full.py                  # Full training pipeline
├── run_pipeline.sh                # End-to-end experiment script
└── requirements.txt               # Python dependencies
```

## Installation

```bash
git clone https://github.com/huyuelin/QMR-Transformer.git
cd QMR-Transformer
pip install -r requirements.txt
```

For Lean 4 formal verification:
```bash
# Install Lean 4: https://leanprover-community.github.io/get_started.html
# Install Mathlib dependencies
```

## Usage

### Training

```bash
# Full training pipeline
python train_full.py --config config.yaml --device cuda

# Run all experiments (compiler stress, same-budget baselines, perturbation sweeps)
python run_full_experiments.py --device cuda
```

### Evaluation

```bash
# Generate result tables
python generate_tables.py --results-dir ./results --output-dir ./tables
```

### Formal Verification

```bash
# Verify Lean 4 proofs
lake build

# Use Seed-Prover for proof search (optional)
python lean_proofs/seed_prover_client.py --lean-file lean_proofs/main_theorems.lean
```

## Theoretical Results

| Theorem | Statement | Verification |
|---------|-----------|:---:|
| Edge-Depth Optimality | $E(L) \geq DL^{1+1/D}$ | Lean 4 |
| Mixed-Radix Coverage | Unique canonical path for every displacement | Lean 4 |
| Reachability-Routability Separation | Query-independent routing yields $\text{Acc} \leq 1/2 + 1/(2L)$ | Lean 4 |
| Softmax Leakage Bound | $P_{\text{path}} \geq \prod_{\ell=1}^{D} (1+\eta_\ell)^{-1}$ | Lean 4 |
| Stability Lift | QMR-Lite guarantees lift to Full++ under bounded perturbation | Lean 4 |

## Citation

```bibtex
@inproceedings{hu2027qmr,
  title={Query-Addressable Mixed-Radix Transformers: Verified Bounds and Efficient Long-Context Retrieval Substrate},
  author={Hu, Yuelin and Yu, Zhenbo and Cheng, Zhengxue and Liu, Wei and Song, Li},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2027}
}
```

## License

This project is released under the MIT License.

## Acknowledgments

- [Seed-Prover](https://github.com/ByteDance-Seed/Seed-Prover) for automated Lean 4 proof drafting
- [Lean 4](https://leanprover-community.github.io/) and Mathlib for formal verification infrastructure
