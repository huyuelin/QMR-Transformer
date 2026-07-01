"""
QMR-Transformers Reproduction Package.

Implements "Query-Addressable Mixed-Radix Transformers:
Verified Bounds and Efficient Long-Context Retrieval Substrate"
(aaai2026.tex).

Usage:
    from models.mixed_radix_generator import MixedRadixGraphGenerator
    from models.compilers import REQMRCompiler
    from models.qmr_architectures import QMRCorePlus

    gen = MixedRadixGraphGenerator()
    radices = gen.compute_radices(L=4096, D=4)
    model = QMRCorePlus(d_model=128, num_layers=4, ...)
"""

__version__ = "0.1.0"
