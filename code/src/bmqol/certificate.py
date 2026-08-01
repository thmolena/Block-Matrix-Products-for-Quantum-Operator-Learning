"""Rank-depth gradient bounds for cosine block products."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from math import factorial
from typing import Iterable, Sequence

import numpy as np

from .krylov import GradientResult, ProbeCompression

Array = np.ndarray


@dataclass(frozen=True)
class GradientCertificate:
    compression_components: Array
    krylov_components: Array
    total_components: Array
    compression_norm: float
    krylov_norm: float
    total_norm: float
    polynomial_degree: int
    radius: float


def preflight_certificate(
    compression: ProbeCompression,
    targets: Iterable[Array],
    times: Iterable[float],
    derivative_norms: Sequence[float],
    *,
    polynomial_degree: int,
    radius: float,
) -> GradientCertificate:
    """A certificate available before the first Hamiltonian application.

    The unknown projected residual is replaced by
    ``||B_r||_2 ||B_r||_F + ||Y_l||_F``, which follows from
    ``||cos(t T)||_2 <= 1`` when the projected spectrum lies in the registered
    interval.  The bound is therefore looser than :func:`certificate_at_state`
    but incurs no candidate factorizations.
    """

    target_tuple = tuple(np.asarray(target, dtype=float) for target in targets)
    time_tuple = tuple(float(value) for value in times)
    compression_components = compression_gradient_bound(
        compression, target_tuple, time_tuple, derivative_norms
    )
    block_factor = (
        compression.compressed_spectral_norm
        * compression.compressed_frobenius_norm
    )
    krylov = np.zeros(len(derivative_norms))
    for parameter, direction_norm in enumerate(derivative_norms):
        total = 0.0
        for target, time_value in zip(target_tuple, time_tuple):
            value_tail, derivative_tail = cosine_tails(
                polynomial_degree, time_value, radius
            )
            response_error = 2.0 * block_factor * value_tail
            derivative_size = block_factor * abs(time_value) * direction_norm
            derivative_error = 2.0 * block_factor * direction_norm * derivative_tail
            residual_upper = block_factor + float(np.linalg.norm(target))
            total += response_error * derivative_size + residual_upper * derivative_error
        krylov[parameter] = total / len(time_tuple)
    total_components = compression_components + krylov
    return GradientCertificate(
        compression_components=compression_components,
        krylov_components=krylov,
        total_components=total_components,
        compression_norm=float(np.linalg.norm(compression_components)),
        krylov_norm=float(np.linalg.norm(krylov)),
        total_norm=float(np.linalg.norm(total_components)),
        polynomial_degree=int(polynomial_degree),
        radius=float(radius),
    )


def _upward(value: Decimal) -> float:
    return float(np.nextafter(float(value), np.inf))


def cosine_tails(degree: int, time_value: float, radius: float) -> tuple[float, float]:
    """Rigorous geometric majorants for value and Frechet cosine tails.

    ``R0`` bounds ``sum_{2k>degree} |t|^(2k) rho^(2k)/(2k)!``.
    ``R1`` bounds the corresponding derivative series with one power of
    ``rho`` removed.  Decimal arithmetic is converted to binary64 outward.
    """

    if degree < 0 or radius < 0.0:
        raise ValueError("degree and radius must be nonnegative")
    if not np.isfinite(time_value) or not np.isfinite(radius):
        raise ValueError("time and radius must be finite")
    first = degree + 1
    if first % 2:
        first += 1
    if time_value == 0.0 or radius == 0.0:
        return 0.0, 0.0
    with localcontext() as context:
        context.prec = 80
        time_decimal = Decimal(str(abs(time_value)))
        radius_decimal = Decimal(str(radius))
        argument = time_decimal * radius_decimal
        value_first = argument**first / Decimal(factorial(first))
        value_ratio = argument**2 / Decimal((first + 1) * (first + 2))
        if value_ratio >= 1:
            return float("inf"), float("inf")
        value_tail = value_first / (Decimal(1) - value_ratio)
        derivative_first = (
            Decimal(first)
            * time_decimal**first
            * radius_decimal ** (first - 1)
            / Decimal(factorial(first))
        )
        derivative_ratio = argument**2 / Decimal(first * (first + 1))
        if derivative_ratio >= 1:
            return float("inf"), float("inf")
        derivative_tail = derivative_first / (Decimal(1) - derivative_ratio)
    return _upward(value_tail), _upward(derivative_tail)


def compression_gradient_bound(
    compression: ProbeCompression,
    targets: Iterable[Array],
    times: Iterable[float],
    derivative_norms: Sequence[float],
) -> Array:
    """A priori gradient loss from replacing ``B`` with ``B_r``."""

    target_tuple = tuple(np.asarray(target, dtype=float) for target in targets)
    time_tuple = tuple(float(value) for value in times)
    if len(target_tuple) != len(time_tuple) or not target_tuple:
        raise ValueError("targets and times must have the same nonzero length")
    alpha = compression.bilinear_compression_factor
    full_factor = (
        compression.full_spectral_norm
        * float(np.linalg.norm(compression.singular_values))
    )
    compressed_factor = (
        compression.compressed_spectral_norm
        * compression.compressed_frobenius_norm
    )
    bounds = np.zeros(len(derivative_norms))
    for parameter, direction_norm in enumerate(derivative_norms):
        total = 0.0
        for target, time_value in zip(target_tuple, time_tuple):
            total += (
                alpha
                * abs(time_value)
                * direction_norm
                * (full_factor + compressed_factor + float(np.linalg.norm(target)))
            )
        bounds[parameter] = total / len(time_tuple)
    return bounds


def certificate_at_state(
    result: GradientResult,
    targets: Iterable[Array],
    times: Iterable[float],
    derivative_norms: Sequence[float],
    *,
    polynomial_degree: int,
    radius: float,
) -> GradientCertificate:
    """Combine probe-rank and Krylov-tail gradient error bounds."""

    target_tuple = tuple(np.asarray(target, dtype=float) for target in targets)
    time_tuple = tuple(float(value) for value in times)
    compression = compression_gradient_bound(
        result.compression, target_tuple, time_tuple, derivative_norms
    )
    block_factor = (
        result.compression.compressed_spectral_norm
        * result.compression.compressed_frobenius_norm
    )
    krylov = np.zeros(len(derivative_norms))
    # A square orthonormal basis is a full-space reduction; the projected
    # matrix and every projected direction are then related to their originals
    # by an orthogonal similarity, so no Krylov truncation remains.
    if result.state.basis.shape[0] == result.state.basis.shape[1]:
        total_components = compression.copy()
        return GradientCertificate(
            compression_components=compression,
            krylov_components=krylov,
            total_components=total_components,
            compression_norm=float(np.linalg.norm(compression)),
            krylov_norm=0.0,
            total_norm=float(np.linalg.norm(total_components)),
            polynomial_degree=int(polynomial_degree),
            radius=float(radius),
        )
    for parameter, direction_norm in enumerate(derivative_norms):
        total = 0.0
        for prediction, target, time_value in zip(
            result.predictions, target_tuple, time_tuple
        ):
            value_tail, derivative_tail = cosine_tails(
                polynomial_degree, time_value, radius
            )
            response_error = 2.0 * block_factor * value_tail
            derivative_size = block_factor * abs(time_value) * direction_norm
            derivative_error = 2.0 * block_factor * direction_norm * derivative_tail
            projected_residual = float(np.linalg.norm(prediction - target))
            total += (
                response_error * derivative_size
                + projected_residual * derivative_error
            )
        krylov[parameter] = total / len(time_tuple)
    total_components = compression + krylov
    return GradientCertificate(
        compression_components=compression,
        krylov_components=krylov,
        total_components=total_components,
        compression_norm=float(np.linalg.norm(compression)),
        krylov_norm=float(np.linalg.norm(krylov)),
        total_norm=float(np.linalg.norm(total_components)),
        polynomial_degree=int(polynomial_degree),
        radius=float(radius),
    )
