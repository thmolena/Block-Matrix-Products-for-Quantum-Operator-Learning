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
            compression, target_tuple, time_tuple, derivative_norms
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
) -> PreflightResult:
    """Select rank and depth before executing one compressed factorization."""

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
    )
    records: list[PreflightRecord] = []
    selected = None
    for depth in depths:
        bound = preflight_certificate(
            compression,
            target_tuple,
            time_tuple,
            family.derivative_norms,
            polynomial_degree=depth,
            radius=family.uniform_radius,
        )
        records.append(PreflightRecord(depth=depth, certificate=bound))
        if bound.total_norm <= tolerance:
            selected = (depth, bound)
            break
    if selected is None:
        raise ValueError("candidate depth grid does not certify the requested tolerance")
    depth, selection_bound = selected
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
