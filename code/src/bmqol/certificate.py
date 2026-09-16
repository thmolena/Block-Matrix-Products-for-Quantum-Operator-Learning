"""Rank-depth gradient bounds for cosine block products."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from math import factorial
from functools import lru_cache
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
    response_components: Array
    derivative_components: Array
    response_norm: float
    derivative_norm: float
    compression_method: str
    tail_method: str


def _inputs(targets, times, derivative_norms, columns):
    target_tuple = tuple(np.asarray(target, dtype=float) for target in targets)
    time_tuple = tuple(float(value) for value in times)
    norms = np.asarray(derivative_norms, dtype=float)
    if len(target_tuple) != len(time_tuple) or not target_tuple:
        raise ValueError("targets and times must have the same nonzero length")
    if norms.ndim != 1 or np.any(~np.isfinite(norms)) or np.any(norms < 0):
        raise ValueError("direction norms must be finite and nonnegative")
    if any(target.shape != (columns, columns) or not np.all(np.isfinite(target))
           for target in target_tuple):
        raise ValueError("targets must be finite square matrices matching the probe columns")
    if not np.all(np.isfinite(time_tuple)):
        raise ValueError("times must be finite")
    return target_tuple, time_tuple, norms


def _assemble(compression, response, derivative, degree, radius,
              compression_method, tail_method):
    krylov = response + derivative
    total = compression + krylov
    return GradientCertificate(
        compression_components=compression,
        krylov_components=krylov,
        total_components=total,
        compression_norm=float(np.linalg.norm(compression)),
        krylov_norm=float(np.linalg.norm(krylov)),
        total_norm=float(np.linalg.norm(total)),
        polynomial_degree=int(degree),
        radius=float(radius),
        response_components=response,
        derivative_components=derivative,
        response_norm=float(np.linalg.norm(response)),
        derivative_norm=float(np.linalg.norm(derivative)),
        compression_method=compression_method,
        tail_method=tail_method,
    )


def _tail_function(method, degree, radius):
    if int(degree) != degree or degree < 1:
        raise ValueError("polynomial degree must be a positive integer")
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and nonnegative")
    if method == "taylor":
        return cosine_tails
    if method == "chebyshev":
        return cosine_chebyshev_tails
    raise ValueError("tail_method must be 'taylor' or 'chebyshev'")


def preflight_certificate(
    compression: ProbeCompression,
    targets: Iterable[Array],
    times: Iterable[float],
    derivative_norms: Sequence[float],
    *,
    polynomial_degree: int,
    radius: float,
    compression_method: str = "loss-aware",
    tail_method: str = "chebyshev",
) -> GradientCertificate:
    """A certificate available before the first Hamiltonian application.

    The unknown projected residual is replaced by
    ``||B_r||_2 ||B_r||_F + ||Y_l||_F``, which follows from
    ``||cos(t T)||_2 <= 1`` when the projected spectrum lies in the registered
    interval.  The bound is therefore looser than :func:`certificate_at_state`
    but incurs no candidate factorizations.
    """

    target_tuple, time_tuple, norms = _inputs(
        targets, times, derivative_norms, compression.full_columns
    )
    tails = _tail_function(tail_method, polynomial_degree, radius)
    compression_components = compression_gradient_bound(
        compression, target_tuple, time_tuple, norms, method=compression_method
    )
    block_factor = (
        compression.compressed_spectral_norm
        * compression.compressed_frobenius_norm
    )
    response = np.zeros(len(norms))
    derivative = np.zeros(len(norms))
    for parameter, direction_norm in enumerate(norms):
        if direction_norm == 0.0 or block_factor == 0.0:
            continue
        for target, time_value in zip(target_tuple, time_tuple):
            value_tail, _ = tails(2 * polynomial_degree - 1, time_value, radius)
            _, derivative_tail = tails(polynomial_degree, time_value, radius)
            response_error = 2.0 * block_factor * value_tail
            derivative_size = block_factor * abs(time_value) * direction_norm
            derivative_error = 2.0 * block_factor * direction_norm * derivative_tail
            residual_upper = block_factor + float(np.linalg.norm(target))
            response[parameter] += response_error * derivative_size / len(time_tuple)
            derivative[parameter] += residual_upper * derivative_error / len(time_tuple)
    return _assemble(
        compression_components, response, derivative, polynomial_degree, radius,
        compression_method, tail_method,
    )


def _upward(value: Decimal) -> float:
    return float(np.nextafter(float(value), np.inf))


@lru_cache(maxsize=4096)
def cosine_tails(degree: int, time_value: float, radius: float) -> tuple[float, float]:
    """Rigorous geometric majorants for value and Frechet cosine tails.

    ``R0`` bounds ``sum_{2k>degree} |t|^(2k) rho^(2k)/(2k)!``.
    ``R1`` bounds the corresponding derivative series with one power of
    ``rho`` removed.  Decimal arithmetic is converted to binary64 outward.
    """

    if int(degree) != degree or degree < 0 or radius < 0.0:
        raise ValueError("degree and radius must be nonnegative")
    if not np.isfinite(time_value) or not np.isfinite(radius):
        raise ValueError("time and radius must be finite")
    degree = int(degree)
    first = degree + 1
    if first % 2:
        first += 1
    if time_value == 0.0 or radius == 0.0:
        return 0.0, 0.0
    with localcontext() as context:
        context.prec = 80
        time_decimal = Decimal.from_float(float(abs(time_value)))
        radius_decimal = Decimal.from_float(float(radius))
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


@lru_cache(maxsize=4096)
def cosine_chebyshev_tails(
    degree: int, time_value: float, radius: float
) -> tuple[float, float]:
    """Value and operator-Frechet bounds for a truncated cosine Chebyshev series.

    For real ``z=t*rho``, the degree-k coefficient (even k>0) has magnitude
    ``2*abs(J_k(z)) <= 2*(abs(z)/2)**k/k!`` (DLMF 10.12.3, 10.14.4).
    Differentiating the Chebyshev recurrence gives, for a symmetric contraction X,

    ``L_Tk(X,E) = U_(k-1)(X) E
                  + 2 sum_(j=1)^(k-1) U_(k-1-j)(X) E T_j(X)``.

    Thus ``||L_Tk(X,E)||_2 <= k**2 ||E||_2``, since ``||T_j(X)||<=1``
    and ``||U_j(X)||<=j+1``. Scaling X=A/rho introduces ``1/rho``.
    This is an operator proof, not an inference from the scalar Markov bound.
    The returned positive geometric majorants do not require Bessel evaluation.
    As with the Taylor certificate, this does not bound the floating-point
    errors in SVD, Krylov orthogonalization, or projected function evaluation.
    """
    if int(degree) != degree or degree < 0 or radius < 0.0:
        raise ValueError("degree and radius must be nonnegative")
    if not np.isfinite(time_value) or not np.isfinite(radius):
        raise ValueError("time and radius must be finite")
    degree = int(degree)
    if time_value == 0.0 or radius == 0.0:
        return 0.0, 0.0
    first = degree + 1
    if first % 2:
        first += 1
    with localcontext() as context:
        context.prec = 80
        rho = Decimal.from_float(float(radius))
        a = Decimal.from_float(float(abs(time_value))) * rho / 2
        coefficient = 2 * a**first / Decimal(factorial(first))
        ratio_value = a**2 / Decimal((first + 1) * (first + 2))
        ratio_derivative = a**2 * Decimal(first + 2) / Decimal(first**2 * (first + 1))
        if ratio_value >= 1:
            value = float("inf")
        else:
            value = _upward(coefficient / (1 - ratio_value))
        if ratio_derivative >= 1:
            derivative = float("inf")
        else:
            derivative = _upward(
                coefficient * Decimal(first**2) / rho / (1 - ratio_derivative)
            )
    return value, derivative


def compression_gradient_bound(
    compression: ProbeCompression,
    targets: Iterable[Array],
    times: Iterable[float],
    derivative_norms: Sequence[float],
    *,
    method: str = "loss-aware",
) -> Array:
    """A priori gradient loss from replacing ``B`` with ``B_r``.

    ``legacy`` uses the original linear-in-discarded-norm product bound.
    ``loss-aware`` takes its minimum with the discarded-block bound. In right
    singular coordinates the retained-retained gradient contribution cancels;
    only two cross blocks and the discarded diagonal block remain. With
    ``u=||C_r||_2 ||C_d||_F`` and ``v=||C_d||_2 ||C_d||_F``, their bound is
    ``|t| ||D|| [2u(u+||Y_rd||_F)+v(v+||Y_dd||_F)]``.
    Targets are symmetrized because the antisymmetric part has zero inner
    product with every real-symmetric response derivative. ``P_d`` includes
    any null columns when B is wide, which preserves validity.
    """
    target_tuple, time_tuple, norms = _inputs(
        targets, times, derivative_norms, compression.full_columns
    )
    if method not in ("legacy", "loss-aware"):
        raise ValueError("method must be 'legacy' or 'loss-aware'")
    alpha = compression.bilinear_compression_factor
    full_factor = (
        compression.full_spectral_norm
        * float(np.linalg.norm(compression.singular_values))
    )
    compressed_factor = (
        compression.compressed_spectral_norm
        * compression.compressed_frobenius_norm
    )
    legacy = sum(alpha * abs(t) * (full_factor + compressed_factor + np.linalg.norm(y))
                 for y, t in zip(target_tuple, time_tuple)) / len(time_tuple)
    if method == "legacy":
        return norms * legacy
    delta = compression.discarded_frobenius
    if delta == 0.0:
        return np.zeros_like(norms)
    singular = compression.singular_values
    sigma_discarded = float(singular[compression.rank]) if compression.rank < len(singular) else 0.0
    u = compression.compressed_spectral_norm * delta
    v = sigma_discarded * delta
    w = compression.mixing
    discarded = np.eye(compression.full_columns) - w @ w.T
    refined = 0.0
    for target, time_value in zip(target_tuple, time_tuple):
        symmetric_target = 0.5 * (target + target.T)
        cross = float(np.linalg.norm(w.T @ symmetric_target @ discarded))
        diagonal = float(np.linalg.norm(discarded @ symmetric_target @ discarded))
        refined += abs(time_value) * (2 * u * (u + cross) + v * (v + diagonal))
    return norms * min(legacy, refined / len(time_tuple))


def certificate_at_state(
    result: GradientResult,
    targets: Iterable[Array],
    times: Iterable[float],
    derivative_norms: Sequence[float],
    *,
    polynomial_degree: int,
    radius: float,
    compression_method: str = "loss-aware",
    tail_method: str = "chebyshev",
) -> GradientCertificate:
    """Combine probe-rank and Krylov-tail gradient error bounds."""

    target_tuple, time_tuple, norms = _inputs(
        targets, times, derivative_norms, result.compression.full_columns
    )
    if len(result.predictions) != len(target_tuple):
        raise ValueError("predictions and targets must have the same length")
    tails = _tail_function(tail_method, polynomial_degree, radius)
    compression = compression_gradient_bound(
        result.compression, target_tuple, time_tuple, norms, method=compression_method
    )
    block_factor = (
        result.compression.compressed_spectral_norm
        * result.compression.compressed_frobenius_norm
    )
    response = np.zeros(len(norms))
    derivative = np.zeros(len(norms))
    # A square orthonormal basis is a full-space reduction; the projected
    # matrix and every projected direction are then related to their originals
    # by an orthogonal similarity, so no Krylov truncation remains.
    if result.state.basis.shape[0] == result.state.basis.shape[1]:
        return _assemble(
            compression, response, derivative, polynomial_degree, radius,
            compression_method, tail_method,
        )
    for parameter, direction_norm in enumerate(norms):
        if direction_norm == 0.0 or block_factor == 0.0:
            continue
        for prediction, target, time_value in zip(
            result.predictions, target_tuple, time_tuple
        ):
            value_tail, _ = tails(2 * polynomial_degree - 1, time_value, radius)
            _, derivative_tail = tails(polynomial_degree, time_value, radius)
            response_error = 2.0 * block_factor * value_tail
            derivative_size = block_factor * abs(time_value) * direction_norm
            derivative_error = 2.0 * block_factor * direction_norm * derivative_tail
            projected_residual = float(np.linalg.norm(prediction - target))
            response[parameter] += response_error * derivative_size / len(time_tuple)
            if projected_residual != 0.0:
                derivative[parameter] += projected_residual * derivative_error / len(time_tuple)
    return _assemble(
        compression, response, derivative, polynomial_degree, radius,
        compression_method, tail_method,
    )
