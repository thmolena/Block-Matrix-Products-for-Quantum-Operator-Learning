"""Rank-depth adaptive evaluation with a predeclared error budget."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .certificate import (
    GradientCertificate,
    certificate_at_state,
    compression_gradient_bound,
    preflight_certificate,
)
from .krylov import (
    GradientResult,
    IncrementalBlockKrylov,
    ProbeCompression,
    probe_compression_path,
    projected_loss_gradient,
)

Array = np.ndarray


@dataclass(frozen=True)
class DepthRecord:
    depth: int
    certificate: GradientCertificate
    gradient: Array
    loss: float
    vector_equivalents: int
    orthogonality_defect: float
    factorization_defect: float


@dataclass(frozen=True)
class AdaptiveResult:
    result: GradientResult
    certificate: GradientCertificate
    rank: int
    depth: int
    tolerance: float
    certified: bool
    rank_budget: float
    records: tuple[DepthRecord, ...]


@dataclass(frozen=True)
class PreflightRecord:
    depth: int
    certificate: GradientCertificate


@dataclass(frozen=True)
class PreflightResult:
    result: GradientResult
    selection_certificate: GradientCertificate
    evaluated_certificate: GradientCertificate
    rank: int
    depth: int
    tolerance: float
    rank_budget: float
    records: tuple[PreflightRecord, ...]


def select_probe_rank(
    probe_block: Array,
    targets: Iterable[Array],
    times: Iterable[float],
    derivative_norms: Sequence[float],
    *,
    tolerance: float,
    rank_budget_fraction: float = 0.20,
    compression_method: str = "loss-aware",
) -> tuple[ProbeCompression, float]:
    """Choose the smallest SVD rank within the compression error budget."""

    if not 0.0 < rank_budget_fraction < 1.0:
        raise ValueError("rank_budget_fraction must lie in (0, 1)")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    target_tuple = tuple(np.asarray(value, dtype=float) for value in targets)
    time_tuple = tuple(float(value) for value in times)
    budget = float(rank_budget_fraction * tolerance)
    for compression in probe_compression_path(probe_block):
        bound = compression_gradient_bound(
            compression, target_tuple, time_tuple, derivative_norms,
            method=compression_method,
        )
        if float(np.linalg.norm(bound)) <= budget:
            return compression, budget
    raise AssertionError("the full probe rank must have zero compression error")


def adaptive_gradient(
    family,
    theta: Array,
    probe_block: Array,
    targets: Iterable[Array],
    times: Iterable[float],
    *,
    tolerance: float,
    candidate_depths: Sequence[int] = (4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24),
    rank_budget_fraction: float = 0.20,
    compression_method: str = "loss-aware",
    tail_method: str = "chebyshev",
) -> AdaptiveResult:
    """Return the first depth whose combined rank-depth bound is feasible."""

    depths = tuple(int(value) for value in candidate_depths)
    if not depths or any(value <= 0 for value in depths):
        raise ValueError("candidate depths must be positive")
    if any(right <= left for left, right in zip(depths, depths[1:])):
        raise ValueError("candidate depths must be strictly increasing")
    target_tuple = tuple(np.asarray(value, dtype=float) for value in targets)
    time_tuple = tuple(float(value) for value in times)
    compression, rank_budget = select_probe_rank(
        probe_block,
        target_tuple,
        time_tuple,
        family.derivative_norms,
        tolerance=tolerance,
        rank_budget_fraction=rank_budget_fraction,
        compression_method=compression_method,
    )
    factorization = IncrementalBlockKrylov(
        family, theta, compression.coordinates
    )
    records: list[DepthRecord] = []
    latest_result = None
    latest_certificate = None
    for depth in depths:
        state = factorization.state(depth)
        latest_result = projected_loss_gradient(
            family, state, compression, target_tuple, time_tuple
        )
        latest_certificate = certificate_at_state(
            latest_result,
            target_tuple,
            time_tuple,
            family.derivative_norms,
            polynomial_degree=state.block_steps,
            radius=family.uniform_radius,
            compression_method=compression_method,
            tail_method=tail_method,
        )
        records.append(
            DepthRecord(
                depth=state.block_steps,
                certificate=latest_certificate,
                gradient=latest_result.gradient.copy(),
                loss=latest_result.loss,
                vector_equivalents=state.hamiltonian_vector_equivalents,
                orthogonality_defect=state.orthogonality_defect,
                factorization_defect=state.factorization_defect,
            )
        )
        if latest_certificate.total_norm <= tolerance:
            return AdaptiveResult(
                result=latest_result,
                certificate=latest_certificate,
                rank=compression.rank,
                depth=state.block_steps,
                tolerance=float(tolerance),
                certified=True,
                rank_budget=rank_budget,
                records=tuple(records),
            )
    assert latest_result is not None and latest_certificate is not None
    return AdaptiveResult(
        result=latest_result,
        certificate=latest_certificate,
        rank=compression.rank,
        depth=latest_result.state.block_steps,
        tolerance=float(tolerance),
        certified=False,
        rank_budget=rank_budget,
        records=tuple(records),
    )


def select_preflight_plan(family, probe_block, targets, times, *, tolerance,
                          candidate_depths, selection="joint", rank_budget_fraction=0.2,
                          compression_method="loss-aware", tail_method="chebyshev"):
    """Minimize (m*r, r, m) over the finite certified grid, or run an ablation.

    The plan is independent of theta and may be reused for fixed observations,
    probes, parameter box, times and tolerance. It is not a runtime optimum.
    """
    depths = tuple(candidate_depths)
    if (not depths or any(int(m) != m or m < 2 for m in depths)
            or any(b <= a for a, b in zip(depths, depths[1:]))):
        raise ValueError("candidate depths must be strictly increasing integers >= 2")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if selection not in ("joint", "rank_first", "full"):
        raise ValueError("unknown selection strategy")
    if not 0 < rank_budget_fraction < 1:
        raise ValueError("rank budget fraction must lie in (0,1)")
    path = probe_compression_path(probe_block)
    if selection == "full":
        path = path[-1:]
    elif selection == "rank_first":
        path = tuple(c for c in path if np.linalg.norm(compression_gradient_bound(
            c, targets, times, family.derivative_norms, method=compression_method))
                     <= rank_budget_fraction * tolerance)[:1]
    candidates = []
    for compression in path:
        if np.linalg.norm(compression_gradient_bound(
                compression, targets, times, family.derivative_norms,
                method=compression_method)) > tolerance:
            continue
        for depth in depths:
            bound = preflight_certificate(compression, targets, times,
                family.derivative_norms, polynomial_degree=depth, radius=family.uniform_radius,
                compression_method=compression_method, tail_method=tail_method)
            if bound.total_norm <= tolerance:
                candidates.append((depth * compression.rank, compression.rank, depth, compression, bound))
                break
    if not candidates:
        raise ValueError("candidate depth grid does not certify the requested tolerance")
    _, _, depth, compression, bound = min(candidates, key=lambda x: x[:3])
    return compression, depth, bound


def preflight_gradient(
    family,
    theta: Array,
    probe_block: Array,
    targets: Iterable[Array],
    times: Iterable[float],
    *,
    tolerance: float,
    candidate_depths: Sequence[int] = (4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32),
    rank_budget_fraction: float = 0.20,
    selection: str = "joint",
    plan=None,
    compression_method: str = "loss-aware",
    tail_method: str = "chebyshev",
) -> PreflightResult:
    """Select rank and depth before executing one compressed factorization."""

    depths = tuple(int(value) for value in candidate_depths)
    if not depths or any(value <= 0 for value in depths):
        raise ValueError("candidate depths must be positive")
    if any(right <= left for left, right in zip(depths, depths[1:])):
        raise ValueError("candidate depths must be strictly increasing")
    target_tuple = tuple(np.asarray(value, dtype=float) for value in targets)
    time_tuple = tuple(float(value) for value in times)
    if plan is None:
        plan = select_preflight_plan(family, probe_block, target_tuple, time_tuple,
            tolerance=tolerance, candidate_depths=depths, selection=selection,
            rank_budget_fraction=rank_budget_fraction,
            compression_method=compression_method, tail_method=tail_method)
    compression, depth, selection_bound = plan
    if selection_bound.total_norm > tolerance:
        raise ValueError("cached plan exceeds the requested tolerance")
    rank_budget = float(rank_budget_fraction * tolerance)
    records = [PreflightRecord(m, preflight_certificate(compression, target_tuple,
        time_tuple, family.derivative_norms, polynomial_degree=m,
        radius=family.uniform_radius, compression_method=compression_method,
        tail_method=tail_method)) for m in depths if m <= depth]
    factorization = IncrementalBlockKrylov(
        family, theta, compression.coordinates
    )
    state = factorization.state(depth)
    result = projected_loss_gradient(
        family, state, compression, target_tuple, time_tuple
    )
    evaluated = certificate_at_state(
        result,
        target_tuple,
        time_tuple,
        family.derivative_norms,
        polynomial_degree=state.block_steps,
        radius=family.uniform_radius,
        compression_method=compression_method,
        tail_method=tail_method,
    )
    return PreflightResult(
        result=result,
        selection_certificate=selection_bound,
        evaluated_certificate=evaluated,
        rank=compression.rank,
        depth=state.block_steps,
        tolerance=float(tolerance),
        rank_budget=rank_budget,
        records=tuple(records),
    )
