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


def test_clustered_cosine_derivative_against_independent_exponential():
    from scipy.linalg import expm_frechet
    rng=np.random.default_rng(114)
    q,_=np.linalg.qr(rng.normal(size=(7,7)))
    a=(q*np.array([.2,.2,.2+1e-13,.2+1e-10,-.1,.8,.81]))@q.T
    d=rng.normal(size=(7,7)); d=(d+d.T)/2
    for t in (.01,2.,12.):
        value,derivative=_cosine_and_frechet(a,d,t)
        expected,expected_derivative=expm_frechet(-1j*t*a,-1j*t*d)
        np.testing.assert_allclose(value,expected.real,atol=3e-14,rtol=3e-13)
        np.testing.assert_allclose(derivative,expected_derivative.real,atol=3e-14,rtol=3e-12)


def test_response_and_derivative_polynomial_exactness_independently():
    from bmqol.krylov import IncrementalBlockKrylov
    rng=np.random.default_rng(413)
    n=30; m=4
    a=rng.normal(size=(n,n)); a=(a+a.T)/30
    d=rng.normal(size=(n,n)); d=(d+d.T)/30
    family=HamiltonianFamily(csr_matrix(a),(csr_matrix(d),),'test',np.linalg.norm(a,np.inf),.5)
    b=rng.normal(size=(n,2))
    state=IncrementalBlockKrylov(family,np.zeros(1),b).state(m)
    v,t,c=state.basis,state.projected,state.input_coordinates
    for k in range(2*m):
        np.testing.assert_allclose(b.T@np.linalg.matrix_power(a,k)@b,
                                  c.T@np.linalg.matrix_power(t,k)@c,atol=2e-12)
    dh=v.T@d@v
    for k in range(1,m+1):
        full=sum(np.linalg.matrix_power(a,u)@d@np.linalg.matrix_power(a,k-1-u) for u in range(k))
        reduced=sum(np.linalg.matrix_power(t,u)@dh@np.linalg.matrix_power(t,k-1-u) for u in range(k))
        np.testing.assert_allclose(b.T@full@b,c.T@reduced@c,atol=2e-12)
    k=m+1
    full=sum(np.linalg.matrix_power(a,u)@d@np.linalg.matrix_power(a,k-1-u) for u in range(k))
    reduced=sum(np.linalg.matrix_power(t,u)@dh@np.linalg.matrix_power(t,k-1-u) for u in range(k))
    assert np.linalg.norm(b.T@full@b-c.T@reduced@c)>1e-8


def test_augmented_reference_and_offdiagonal_projected_gradient():
    from bmqol.experiment import augmented_reference
    from bmqol.krylov import evaluate_at_depth
    from scipy.linalg import expm_frechet
    rng=np.random.default_rng(933)
    raw=rng.normal(size=(16,16)); raw=(raw+raw.T)/40
    d=rng.normal(size=(16,16)); d=(d+d.T)/70
    family=HamiltonianFamily(csr_matrix(raw),(csr_matrix(d),),'offdiagonal',np.linalg.norm(raw,np.inf),.5)
    b=rng.normal(size=(16,3)); theta=np.array([.13]); times=(.5,3.)
    targets=[np.eye(3)]*2
    predictions,derivatives,g,_=augmented_reference(family,theta,b,targets,times)
    for ell,t in enumerate(times):
        value,derivative=expm_frechet(-1j*t*family.dense(theta),-1j*t*d)
        np.testing.assert_allclose(predictions[ell],b.T@value.real@b,atol=1e-12)
        np.testing.assert_allclose(derivatives[ell][0],b.T@derivative.real@b,atol=1e-12)
    result=evaluate_at_depth(family,theta,b,targets,times,rank=3,block_steps=8)
    np.testing.assert_allclose(result.gradient,g,atol=2e-11)


def test_joint_plan_is_grid_work_minimum():
    from bmqol.adaptive import select_preflight_plan
    from bmqol.krylov import probe_compression_path
    from bmqol.certificate import preflight_certificate
    family=small_family()
    rng=np.random.default_rng(55)
    u,_=np.linalg.qr(rng.normal(size=(18,5)))
    b=u*np.array([1,.1,.01,1e-5,1e-8])
    targets=(np.eye(5),); times=(2.,); depths=tuple(range(2,20,2))
    selected,depth,bound=select_preflight_plan(family,b,targets,times,tolerance=1e-3,candidate_depths=depths)
    feasible=[]
    for compression in probe_compression_path(b):
        for m in depths:
            certificate=preflight_certificate(compression,targets,times,family.derivative_norms,
                                               polynomial_degree=m,radius=family.uniform_radius)
            if certificate.total_norm<=1e-3:
                feasible.append(m*compression.rank)
    assert selected.rank*depth==min(feasible)
    assert bound.total_norm<=1e-3


def test_scalar_polarization_against_independent_block_response():
    from bmqol.experiment import augmented_reference, polarization_gradient
    family=small_family(45)
    rng=np.random.default_rng(242)
    b=rng.normal(size=(family.n,3))
    targets=[np.eye(3),np.zeros((3,3))]
    times=(1.,3.)
    theta=np.array([.1,-.15])
    _,_,expected,_=augmented_reference(family,theta,b,targets,times)
    actual,work=polarization_gradient(family,theta,b,targets,times,18)
    np.testing.assert_allclose(actual,expected,atol=2e-11)
    assert work==6*18


def test_semantic_digest_retains_physical_time_and_excludes_measurements():
    from bmqol.experiment import semantic_digest
    import copy
    payload={'times':[2.,4.], 'frontier':[{'depth':4,'error':.00012,
              'timing':{'seconds':[1.,2.],'median_seconds':1.5}}]}
    digest=semantic_digest(payload)
    updated=copy.deepcopy(payload)
    updated['frontier'][0]['timing']['median_seconds']=900.
    assert semantic_digest(updated)==digest
    updated['times'][0]=3.
    assert semantic_digest(updated)!=digest


def test_chebyshev_operator_derivative_bound_noncommuting_and_endpoint():
    """Check the operator inequality without using a scalar derivative bound."""
    from numpy.polynomial.chebyshev import cheb2poly
    rng = np.random.default_rng(705)
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    matrix = (q * np.array([-1., -1., -.4, .2, 1., 1.])) @ q.T
    direction = rng.normal(size=(6, 6))
    direction = (direction + direction.T) / 2
    for degree in range(1, 13):
        coefficients = cheb2poly(np.eye(1, degree + 1, degree).ravel())
        derivative = np.zeros_like(matrix)
        for power, coefficient in enumerate(coefficients):
            for split in range(power):
                derivative += (coefficient * np.linalg.matrix_power(matrix, split)
                               @ direction @ np.linalg.matrix_power(matrix, power - 1 - split))
        assert np.linalg.norm(derivative, 2) <= degree**2 * np.linalg.norm(direction, 2) * (1 + 1e-11)
        # At X=I, the exact polynomial derivative is k^2 E: the constant is sharp.
        endpoint_factor = sum(power * coefficient for power, coefficient in enumerate(coefficients))
        assert abs(endpoint_factor - degree**2) < 1e-10


def test_chebyshev_tails_cover_independent_matrix_frechet_remainders():
    from bmqol.certificate import cosine_chebyshev_tails
    from scipy.linalg import expm_frechet
    from scipy.special import jv
    rng = np.random.default_rng(711)
    q, _ = np.linalg.qr(rng.normal(size=(7, 7)))
    radius = .98
    x = (q * np.array([-1., -.7, -.1, .2, .2, .8, 1.])) @ q.T
    direction = rng.normal(size=(7, 7))
    direction = (direction + direction.T) / 2
    for time in (.5, 3., 8.):
        exact_value, exact_derivative = expm_frechet(-1j * time * radius * x,
                                                    -1j * time * direction)
        for degree in (0, 2, 4, 8):
            value = jv(0, time * radius) * np.eye(7)
            derivative = np.zeros((7, 7))
            previous, current = np.eye(7), x
            previous_d, current_d = np.zeros((7, 7)), direction / radius
            for k in range(1, degree + 1):
                if k % 2 == 0:
                    coefficient = 2 * (-1)**(k // 2) * jv(k, time * radius)
                    value += coefficient * current
                    derivative += coefficient * current_d
                previous, current, previous_d, current_d = (
                    current, 2 * x @ current - previous, current_d,
                    2 * (direction / radius @ current + x @ current_d) - previous_d,
                )
            value_bound, derivative_bound = cosine_chebyshev_tails(degree, time, radius)
            assert np.linalg.norm(exact_value.real - value, 2) <= value_bound + 3e-14
            assert np.linalg.norm(exact_derivative.real - derivative, 2) <= (
                derivative_bound * np.linalg.norm(direction, 2) + 3e-13
            )
    # Representative certificate regime has much smaller tails than Taylor.
    old = cosine_tails(16, 6., radius)
    new = cosine_chebyshev_tails(16, 6., radius)
    assert new[0] < old[0] / 1000
    assert new[1] < old[1] / 1000


def test_discarded_block_bound_against_independent_full_and_compressed_gradients():
    from bmqol.certificate import compression_gradient_bound
    from bmqol.krylov import probe_compression_path
    from scipy.linalg import expm_frechet
    rng = np.random.default_rng(743)
    n, columns = 17, 7
    u, _ = np.linalg.qr(rng.normal(size=(n, columns)))
    w, _ = np.linalg.qr(rng.normal(size=(columns, columns)))
    probes = (u * np.geomspace(1, .003, columns)) @ w.T
    matrix = rng.normal(size=(n, n))
    matrix = (matrix + matrix.T) / (2 * n)
    directions = []
    for _ in range(3):
        raw = rng.normal(size=(n, n))
        directions.append((raw + raw.T) / (2 * n))
    times = (.8, 3.2)
    norms = [np.linalg.norm(d, 2) for d in directions]
    references = [[expm_frechet(-1j * time * matrix, -1j * time * d)
                   for d in directions] for time in times]
    for noise in (0., .01, 1.):
        targets = tuple(probes.T @ row[0][0].real @ probes
                        + noise * rng.normal(size=(columns, columns)) for row in references)
        for compression in probe_compression_path(probes):
            reduced = compression.coordinates @ compression.mixing.T
            difference = np.zeros(3)
            for target, row in zip(targets, references):
                full_response = probes.T @ row[0][0].real @ probes
                reduced_response = reduced.T @ row[0][0].real @ reduced
                for j, (_, derivative) in enumerate(row):
                    difference[j] += (
                        np.sum((full_response - target) * (probes.T @ derivative.real @ probes))
                        - np.sum((reduced_response - target) * (reduced.T @ derivative.real @ reduced))
                    ) / len(times)
            bound = compression_gradient_bound(compression, targets, times, norms)
            old = compression_gradient_bound(compression, targets, times, norms, method="legacy")
            assert np.all(np.abs(difference) <= bound + 3e-15)
            assert np.all(bound <= old)


def test_loss_aware_compression_bound_scales_quadratically_for_consistent_targets():
    from bmqol.certificate import compression_gradient_bound
    from bmqol.krylov import compress_probe
    bounds, legacy = [], []
    for discarded in (1e-2, 1e-3):
        probes = np.diag([1., discarded])
        target = probes.T @ np.diag(np.cos(np.array([.2, .7]))) @ probes
        compression = compress_probe(probes, 1)
        bounds.append(compression_gradient_bound(compression, [target], [1.], [1.])[0])
        legacy.append(compression_gradient_bound(compression, [target], [1.], [1.], method="legacy")[0])
    assert 99 < bounds[0] / bounds[1] < 101
    assert 9 < legacy[0] / legacy[1] < 11


def test_certificate_tail_modes_scalar_zero_and_rank_zero_edges():
    from dataclasses import replace
    import pytest
    from bmqol.certificate import (compression_gradient_bound, cosine_chebyshev_tails,
                                   preflight_certificate)
    from bmqol.krylov import compress_probe
    scalar = compress_probe(np.array([[2.]]), 1)
    for method in ("chebyshev", "taylor"):
        bound = preflight_certificate(scalar, [np.array([[3.]])], [0.], [1.],
                                      polynomial_degree=1, radius=0., tail_method=method)
        assert bound.total_norm == 0
        np.testing.assert_array_equal(bound.krylov_components,
                                      bound.response_components + bound.derivative_components)
        zero_direction = preflight_certificate(scalar, [np.array([[3.]])], [100.], [0.],
                                                polynomial_degree=1, radius=1., tail_method=method)
        assert zero_direction.total_norm == 0
    # The bounding routine supports a rank-zero candidate even though the
    # current Krylov selector deliberately searches positive ranks only.
    zero_rank = replace(scalar, rank=0, coordinates=np.empty((1, 0)), mixing=np.empty((1, 0)),
                        compressed_spectral_norm=0., compressed_frobenius_norm=0.,
                        discarded_frobenius=2.)
    bound = compression_gradient_bound(zero_rank, [np.array([[3.]])], [1.], [1.])
    assert bound[0] == 28
    assert cosine_chebyshev_tails(0, 1., 0.) == (0., 0.)
    with pytest.raises(ValueError, match="nonnegative"):
        compression_gradient_bound(scalar, [np.eye(1)], [1.], [-1.])
    with pytest.raises(ValueError, match="tail_method"):
        preflight_certificate(scalar, [np.eye(1)], [1.], [1.], polynomial_degree=1,
                              radius=1., tail_method="unproved")


def test_refined_and_legacy_certificates_cover_nonfullspace_gradient_errors():
    from bmqol.certificate import preflight_certificate, certificate_at_state
    from bmqol.krylov import evaluate_at_depth
    from scipy.linalg import expm_frechet
    rng = np.random.default_rng(751)
    n, columns = 60, 5
    raw = rng.normal(size=(n, n))
    a = (raw + raw.T) / (4 * n)
    raw = rng.normal(size=(n, n))
    d = (raw + raw.T) / (8 * n)
    family = HamiltonianFamily(csr_matrix(a), (csr_matrix(d),), "certificate-check",
                               np.linalg.norm(a, np.inf), .5)
    u, _ = np.linalg.qr(rng.normal(size=(n, columns)))
    b = u * np.geomspace(1., 1e-4, columns)
    targets = (rng.normal(size=(columns, columns)) / 100,)
    time = 3.
    value, derivative = expm_frechet(-1j * time * a, -1j * time * d)
    exact = np.sum((b.T @ value.real @ b - targets[0]) * (b.T @ derivative.real @ b))
    result = evaluate_at_depth(family, np.zeros(1), b, targets, [time], rank=3, block_steps=8)
    assert result.state.basis.shape[1] < n
    for compression_method in ("legacy", "loss-aware"):
        for tail_method in ("taylor", "chebyshev"):
            kwargs = dict(polynomial_degree=8, radius=family.uniform_radius,
                          compression_method=compression_method, tail_method=tail_method)
            preflight = preflight_certificate(result.compression, targets, [time],
                                              family.derivative_norms, **kwargs)
            evaluated = certificate_at_state(result, targets, [time], family.derivative_norms, **kwargs)
            error = abs(exact - result.gradient[0])
            assert error <= evaluated.total_norm
            assert evaluated.total_norm <= preflight.total_norm
            np.testing.assert_allclose(preflight.total_components, preflight.compression_components
                                       + preflight.response_components + preflight.derivative_components)


def test_projected_adjoint_matches_independent_exponential_with_offdiagonal_directions():
    from bmqol.experiment import projected_adjoint_gradient
    from bmqol.krylov import compress_probe, IncrementalBlockKrylov
    from scipy.linalg import expm_frechet
    rng = np.random.default_rng(757)
    n, columns = 25, 4
    raw = rng.normal(size=(n, n))
    a = (raw + raw.T) / (4 * n)
    directions = []
    for _ in range(3):
        raw = rng.normal(size=(n, n))
        directions.append(csr_matrix((raw + raw.T) / (8 * n)))
    family = HamiltonianFamily(csr_matrix(a), tuple(directions), "adjoint-check",
                               np.linalg.norm(a, np.inf), .5)
    b = rng.normal(size=(n, columns)) / np.sqrt(n)
    theta = np.array([.1, -.2, .05])
    targets = tuple(rng.normal(size=(columns, columns)) for _ in range(2))
    times = (.5, 4.)
    for rank in (2, 4):
        compression = compress_probe(b, rank)
        state = IncrementalBlockKrylov(family, theta, compression.coordinates).state(3)
        assert state.basis.shape[1] < n
        coordinates = state.input_coordinates @ compression.mixing.T
        expected = np.zeros(3)
        for target, time in zip(targets, times):
            for j, direction in enumerate(directions):
                value, derivative = expm_frechet(-1j * time * state.projected,
                    -1j * time * (state.basis.T @ direction @ state.basis))
                expected[j] += np.sum((coordinates.T @ value.real @ coordinates - target)
                                     * (coordinates.T @ derivative.real @ coordinates)) / len(times)
        actual = projected_adjoint_gradient(family, state, compression, targets, times)
        np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=3e-12)


def test_incremental_adjoint_counts_only_new_operator_actions():
    from bmqol.experiment import incremental_adjoint_gradient
    from bmqol.krylov import IncrementalBlockKrylov, compress_probe
    rng = np.random.default_rng(761)
    family = small_family(61)
    b = rng.normal(size=(family.n, 2))
    theta = np.array([.1, -.2])
    depths = (1, 2, 3)
    result = incremental_adjoint_gradient(family, theta, b, [np.eye(2)], [6.],
                                          rank=2, tolerance=1e-30, depths=depths)
    assert not result["stopped_by_agreement"]
    expected = IncrementalBlockKrylov(family, theta, compress_probe(b, 2).coordinates).state(3)
    assert result["work"] == expected.hamiltonian_vector_equivalents == 6
    assert result["operator_block_actions"] == expected.operator_block_actions == 3
    assert sum(row["incremental_work"] for row in result["checkpoints"]) == result["work"]
    assert [row["cumulative_work"] for row in result["checkpoints"]] == [2, 4, 6]


def test_counted_learning_reference_matches_explicit_augmented_actions():
    from bmqol.experiment import _counted_learning_reference, augmented_reference
    family = small_family(83)
    rng = np.random.default_rng(769)
    probes = rng.normal(size=(family.n, 3)) / np.sqrt(family.n)
    theta = np.array([.1, -.2])
    times = (.7, 2.)
    targets = tuple(rng.normal(size=(3, 3)) for _ in times)
    expected = augmented_reference(family, theta, probes, targets, times)
    actual = _counted_learning_reference(family, theta, probes, targets, times, derivatives=True)
    for observed, reference in zip(actual[:4], expected):
        np.testing.assert_allclose(observed, reference, rtol=3e-12, atol=3e-14)
    counts = actual[-1]
    assert counts["hamiltonian_vectors"] > 0
    assert counts["direction_vectors"] > 0
    assert counts["exponential_actions"] == len(times)
    assert counts["hamiltonian_vectors"] % (family.num_parameters + 1) == 0
    forward = _counted_learning_reference(family, theta, probes, targets, times)
    np.testing.assert_allclose(forward[0], expected[0], rtol=3e-12, atol=3e-14)
    np.testing.assert_allclose(forward[3], expected[3], rtol=3e-12, atol=3e-14)
    assert forward[2] is None
    assert forward[-1]["hamiltonian_vectors"] > 0
    assert forward[-1]["direction_vectors"] == 0


def test_certified_learning_descent_stationarity_and_audit_accounting():
    from bmqol.experiment import (_certified_learning_fit, _scalable_learning_problem,
                                  augmented_reference)
    from threadpoolctl import threadpool_limits
    family, probes = _scalable_learning_problem(256)
    truth = np.array([.22, -.18, .12])
    initial = np.zeros(3)
    times, held_times = (1., 2., 3.), (1.5, 2.5, 4.)
    zeros = [np.zeros((probes.shape[1], probes.shape[1]))] * len(times)
    tolerance = 1e-5
    with threadpool_limits(limits=1):
        targets = augmented_reference(family, truth, probes, zeros, times)[0]
        held = augmented_reference(family, truth, probes, zeros, held_times)[0]
        result = _certified_learning_fit(family, probes, targets, times, held, held_times,
                                        truth, initial, "joint", stationarity_tolerance=tolerance)
        # Re-evaluate the final KKT residual independently of the stored audit.
        estimate = np.asarray(result["estimate"])
        gradient = augmented_reference(family, estimate, probes, targets, times)[2]
    residual = estimate - np.clip(estimate - gradient, -.5, .5)
    assert result["status"] == "projected_stationarity"
    assert np.linalg.norm(residual) <= tolerance + 1e-12
    np.testing.assert_allclose(result["projected_gradient_norm"], np.linalg.norm(residual), atol=1e-12)
    assert result["history"]
    assert any(row["rank"] < probes.shape[1] and row["basis_dimension"] < family.n
               for row in result["history"])
    for row in result["evaluations"]:
        # This is a regression allowance for binary64, not an interval proof.
        roundoff = 5e-12 * (1 + np.linalg.norm(row["gradient"]))
        assert row["independent_gradient_error"] <= row["gradient_bound"] + roundoff
        assert row["independent_gradient_error"] <= row["preflight_gradient_bound"] + roundoff
    previous = initial
    previous_loss = float("inf")
    for row in result["history"]:
        evaluated = result["evaluations"][row["evaluation_index"]]
        np.testing.assert_array_equal(row["evaluation_theta"], evaluated["theta"])
        np.testing.assert_allclose(row["evaluation_theta"], previous, atol=1e-14)
        np.testing.assert_allclose(row["theta"],
            previous + row["step"] * np.asarray(row["direction"]), atol=1e-14)
        assert row["directional_derivative_upper"] < 0
        assert row["independent_directional_derivative"] < 0
        assert row["loss"] < previous_loss
        previous = np.asarray(row["theta"])
        previous_loss = row["loss"]
    np.testing.assert_array_equal(previous, estimate)
    assert 0 < result["optimization_reference_work"] < result["reference_work"]
    assert 0 <= result["optimization_reference_direction_work"] < result["reference_direction_work"]
    assert 0 < result["optimization_reference_exponential_actions"] < result["reference_exponential_actions"]
    assert result["work"] == result["evaluations"][-1]["work"] > 0
    assert result["direction_work"] == family.num_parameters * sum(
        row["basis_dimension"] for row in result["evaluations"])
