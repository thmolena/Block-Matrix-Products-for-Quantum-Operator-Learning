from __future__ import annotations

import math

import numpy as np
from scipy.sparse import csr_matrix

from bmqol.adaptive import adaptive_gradient, preflight_gradient
from bmqol.certificate import cosine_tails
from bmqol.krylov import _cosine_and_frechet
from bmqol.model import HamiltonianFamily


def small_family(seed: int = 7) -> HamiltonianFamily:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(18, 18))
    raw = 0.5 * (raw + raw.T)
    row_sum = np.max(np.sum(np.abs(raw), axis=1))
    base = csr_matrix(0.65 * raw / row_sum)
    grid = np.linspace(0.0, 1.0, 18)
    fields = (
        0.10 * np.exp(-((grid - 0.3) / 0.18) ** 2),
        0.10 * np.exp(-((grid - 0.7) / 0.18) ** 2),
    )
    return HamiltonianFamily(base, fields, "synthetic-test", 0.65, 0.5)


def dense_products_gradient(family, theta, probes, targets, times):
    matrix = family.dense(theta)
    gradient = np.zeros(family.num_parameters)
    predictions = []
    for target, time_value in zip(targets, times):
        derivatives = []
        cosine = None
        for field in family.fields:
            cosine, frechet = _cosine_and_frechet(
                matrix, np.diag(field), time_value
            )
            derivatives.append(probes.T @ frechet @ probes)
        prediction = probes.T @ cosine @ probes
        predictions.append(prediction)
        error = prediction - target
        for parameter, derivative in enumerate(derivatives):
            gradient[parameter] += np.sum(error * derivative) / len(times)
    return tuple(predictions), gradient


def test_uniform_parameter_box_encloses_eigenvalues() -> None:
    family = small_family()
    rng = np.random.default_rng(10)
    for _ in range(20):
        theta = rng.uniform(-0.5, 0.5, size=family.num_parameters)
        observed = np.linalg.norm(family.dense(theta), ord=2)
        assert observed <= family.uniform_radius + 1.0e-12
    assert family.uniform_radius < 1.0


def test_geometric_tails_dominate_direct_series() -> None:
    degree, time_value, radius = 8, 3.0, 0.98
    value, derivative = cosine_tails(degree, time_value, radius)
    indices = range(10, 80, 2)
    direct_value = sum(
        abs(time_value) ** k * radius**k / math.factorial(k) for k in indices
    )
    direct_derivative = sum(
        k * abs(time_value) ** k * radius ** (k - 1) / math.factorial(k)
        for k in indices
    )
    assert value >= direct_value
    assert derivative >= direct_derivative


def test_rank_depth_certificate_covers_dense_gradient_error() -> None:
    family = small_family()
    grid = np.linspace(0.0, 1.0, family.n)
    probes = np.column_stack(
        [np.exp(-0.5 * ((grid - center) / 0.22) ** 2) for center in np.linspace(0.43, 0.57, 6)]
    )
    probes /= np.linalg.norm(probes, axis=0)[None, :]
    times = (0.7, 1.4, 2.1)
    theta_true = np.array([0.28, -0.17])
    theta = np.array([0.05, -0.02])
    zero_targets = tuple(np.zeros((6, 6)) for _ in times)
    targets, _ = dense_products_gradient(
        family, theta_true, probes, zero_targets, times
    )
    _, dense_gradient = dense_products_gradient(
        family, theta, probes, targets, times
    )
    adaptive = adaptive_gradient(
        family,
        theta,
        probes,
        targets,
        times,
        tolerance=5.0e-4,
        candidate_depths=(4, 6, 8, 10, 12, 14),
    )
    actual = np.linalg.norm(adaptive.result.gradient - dense_gradient)
    assert adaptive.certified
    assert actual <= adaptive.certificate.total_norm


def test_invalid_candidate_grid_is_rejected() -> None:
    family = small_family()
    probes = np.eye(family.n, 3)
    targets = (np.eye(3),)
    try:
        adaptive_gradient(
            family,
            np.zeros(2),
            probes,
            targets,
            (1.0,),
            tolerance=1.0e-3,
            candidate_depths=(4, 4),
        )
    except ValueError as error:
        assert "strictly increasing" in str(error)
    else:
        raise AssertionError("invalid depth grid was accepted")


def test_preflight_selection_uses_one_factorization_and_covers_error() -> None:
    family = small_family()
    grid = np.linspace(0.0, 1.0, family.n)
    probes = np.column_stack(
        [np.exp(-0.5 * ((grid - center) / 0.22) ** 2) for center in np.linspace(0.43, 0.57, 6)]
    )
    probes /= np.linalg.norm(probes, axis=0)[None, :]
    times = (0.7, 1.4, 2.1)
    zero_targets = tuple(np.zeros((6, 6)) for _ in times)
    targets, _ = dense_products_gradient(
        family, np.array([0.28, -0.17]), probes, zero_targets, times
    )
    _, dense_gradient = dense_products_gradient(
        family, np.array([0.05, -0.02]), probes, targets, times
    )
    result = preflight_gradient(
        family,
        np.array([0.05, -0.02]),
        probes,
        targets,
        times,
        tolerance=5.0e-4,
        candidate_depths=(4, 6, 8, 10, 12, 14),
    )
    actual = np.linalg.norm(result.result.gradient - dense_gradient)
    assert result.selection_certificate.total_norm <= result.tolerance
    assert actual <= result.selection_certificate.total_norm
    assert result.result.state.operator_block_actions == result.depth
