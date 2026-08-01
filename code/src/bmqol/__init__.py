"""Rank-depth adaptive block matrix products for quantum operator learning."""

from .certificate import (
    certificate_at_state,
    compression_gradient_bound,
    cosine_tails,
)
from .adaptive import adaptive_gradient, preflight_gradient, select_probe_rank
from .krylov import IncrementalBlockKrylov, projected_loss_gradient
from .model import HamiltonianFamily, build_family, correlated_probe_block

__all__ = [
    "HamiltonianFamily",
    "IncrementalBlockKrylov",
    "build_family",
    "adaptive_gradient",
    "certificate_at_state",
    "compression_gradient_bound",
    "correlated_probe_block",
    "cosine_tails",
    "projected_loss_gradient",
    "preflight_gradient",
    "select_probe_rank",
]

__version__ = "1.0.0"
