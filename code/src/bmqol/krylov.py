"""Incremental block Krylov products and projected Frechet gradients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class ProbeCompression:
    """Rank-r factorization ``B_r = coordinates @ mixing.T``."""

    coordinates: Array
    mixing: Array
    rank: int
    full_columns: int
    discarded_frobenius: float
    full_spectral_norm: float
    compressed_spectral_norm: float
    compressed_frobenius_norm: float
    singular_values: Array

    @property
    def bilinear_compression_factor(self) -> float:
        delta = self.discarded_frobenius
        return float(2.0 * self.full_spectral_norm * delta)


@dataclass(frozen=True)
class KrylovState:
    basis: Array
    projected: Array
    input_coordinates: Array
    block_steps: int
    operator_block_actions: int
    hamiltonian_vector_equivalents: int
    orthogonality_defect: float
    factorization_defect: float


@dataclass(frozen=True)
class GradientResult:
    loss: float
    gradient: Array
    predictions: tuple[Array, ...]
    derivatives: tuple[tuple[Array, ...], ...]
    state: KrylovState
    compression: ProbeCompression


def compress_probe(block: Array, rank: int) -> ProbeCompression:
    block = np.asarray(block, dtype=float)
    if block.ndim != 2:
        raise ValueError("probe block must be two dimensional")
    compressions = probe_compression_path(block)
    if not 1 <= rank <= len(compressions):
        raise ValueError("rank is outside the probe-block range")
    return compressions[rank - 1]


def probe_compression_path(block: Array) -> tuple[ProbeCompression, ...]:
    """Compute every nested truncated SVD after one factorization of ``B``."""

    block = np.asarray(block, dtype=float)
    if block.ndim != 2:
        raise ValueError("probe block must be two dimensional")
    u, singular, vt = np.linalg.svd(block, full_matrices=False)
    records = []
    for rank in range(1, singular.size + 1):
        records.append(
            ProbeCompression(
                coordinates=u[:, :rank] * singular[:rank][None, :],
                mixing=vt[:rank, :].T,
                rank=rank,
                full_columns=block.shape[1],
                discarded_frobenius=float(np.linalg.norm(singular[rank:])),
                full_spectral_norm=float(singular[0]),
                compressed_spectral_norm=float(singular[0]),
                compressed_frobenius_norm=float(np.linalg.norm(singular[:rank])),
                singular_values=singular,
            )
        )
    return tuple(records)


def _orthogonal_block(candidate: Array, blocks: list[Array], tolerance: float) -> Array:
    work = np.array(candidate, dtype=float, copy=True)
    for _ in range(2):
        for block in blocks:
            work -= block @ (block.T @ work)
    q, r = np.linalg.qr(work, mode="reduced")
    scale = max(float(np.linalg.norm(candidate)), np.finfo(float).tiny)
    keep = np.abs(np.diag(r)) > tolerance * scale
    return q[:, keep]


class IncrementalBlockKrylov:
    """Fully reorthogonalized block Krylov factorization extended in place."""

    def __init__(self, family, theta: Array, coordinates: Array, tolerance: float = 1e-12):
        self.family = family
        self.theta = family.validate_theta(theta)
        self.coordinates = np.asarray(coordinates, dtype=float)
        q0, r0 = np.linalg.qr(self.coordinates, mode="reduced")
        keep = np.abs(np.diag(r0)) > tolerance * max(
            float(np.linalg.norm(self.coordinates)), np.finfo(float).tiny
        )
        q0 = q0[:, keep]
        if q0.shape[1] != self.coordinates.shape[1]:
            raise ValueError("compressed probe coordinates are rank deficient")
        self.tolerance = float(tolerance)
        self.blocks: list[Array] = [q0]
        self.applied: list[Array] = []
        self.block_actions = 0
        self.vector_equivalents = 0

    def _apply_unseen(self) -> None:
        while len(self.applied) < len(self.blocks):
            block = self.blocks[len(self.applied)]
            self.applied.append(self.family.apply(self.theta, block))
            self.block_actions += 1
            self.vector_equivalents += block.shape[1]

    def state(self, block_steps: int) -> KrylovState:
        if block_steps < len(self.blocks):
            raise ValueError("an incremental factorization cannot move backward")
        while len(self.blocks) < block_steps:
            self._apply_unseen()
            new = _orthogonal_block(self.applied[-1], self.blocks, self.tolerance)
            if new.shape[1] == 0:
                break
            self.blocks.append(new)
        self._apply_unseen()
        basis = np.column_stack(self.blocks)
        applied = np.column_stack(self.applied)
        projected = basis.T @ applied
        projected = 0.5 * (projected + projected.T)
        identity = np.eye(basis.shape[1])
        orthogonality = float(np.linalg.norm(basis.T @ basis - identity, ord=2))
        residual = applied - basis @ projected
        defect = float(
            np.linalg.norm(residual)
            / max(np.linalg.norm(applied), np.finfo(float).tiny)
        )
        return KrylovState(
            basis=basis,
            projected=projected,
            input_coordinates=basis.T @ self.coordinates,
            block_steps=len(self.blocks),
            operator_block_actions=self.block_actions,
            hamiltonian_vector_equivalents=self.vector_equivalents,
            orthogonality_defect=orthogonality,
            factorization_defect=defect,
        )


def cosine_loewner(eigenvalues: Array, time_value: float) -> Array:
    """Cancellation-free cosine divided differences, including repeated roots."""
    left, right = eigenvalues[:, None], eigenvalues[None, :]
    return (-time_value * np.sin(time_value * (left + right) / 2)
            * np.sinc(time_value * (left - right) / (2 * np.pi)))


def _cosine_and_frechet(projected: Array, direction: Array, time_value: float) -> tuple[Array, Array]:
    eigenvalues, eigenvectors = np.linalg.eigh(projected)
    values = np.cos(time_value * eigenvalues)
    cosine = (eigenvectors * values[None, :]) @ eigenvectors.T
    loewner = cosine_loewner(eigenvalues, time_value)
    rotated_direction = eigenvectors.T @ direction @ eigenvectors
    frechet = eigenvectors @ (loewner * rotated_direction) @ eigenvectors.T
    return cosine, frechet


def _matrix_cosine(projected: Array, time_value: float) -> Array:
    eigenvalues, eigenvectors = np.linalg.eigh(projected)
    values = np.cos(time_value * eigenvalues)
    return (eigenvectors * values[None, :]) @ eigenvectors.T


def projected_predictions(
    state: KrylovState,
    compression: ProbeCompression,
    times: Iterable[float],
) -> tuple[Array, ...]:
    """Evaluate lifted block products without forming parameter derivatives."""

    predictions = []
    coordinates = state.input_coordinates
    mixing = compression.mixing
    for time_value in times:
        cosine = _matrix_cosine(state.projected, float(time_value))
        core = coordinates.T @ cosine @ coordinates
        predictions.append(mixing @ core @ mixing.T)
    return tuple(predictions)


def loss_only(
    family,
    theta: Array,
    probe_block: Array,
    targets: Iterable[Array],
    times: Iterable[float],
    *,
    rank: int,
    block_steps: int,
) -> tuple[float, KrylovState]:
    """Recompute a projected loss for finite-difference baselines."""

    target_tuple = tuple(np.asarray(target, dtype=float) for target in targets)
    time_tuple = tuple(float(value) for value in times)
    compression = compress_probe(probe_block, rank)
    factorization = IncrementalBlockKrylov(family, theta, compression.coordinates)
    state = factorization.state(block_steps)
    predictions = projected_predictions(state, compression, time_tuple)
    loss = sum(
        0.5 * float(np.linalg.norm(prediction - target) ** 2) / len(time_tuple)
        for prediction, target in zip(predictions, target_tuple)
    )
    return loss, state


def projected_loss_gradient(
    family,
    state: KrylovState,
    compression: ProbeCompression,
    targets: Iterable[Array],
    times: Iterable[float],
) -> GradientResult:
    """Evaluate lifted block products and all parameter derivatives."""

    target_tuple = tuple(np.asarray(target, dtype=float) for target in targets)
    time_tuple = tuple(float(value) for value in times)
    if len(target_tuple) != len(time_tuple) or not target_tuple:
        raise ValueError("targets and times must have the same nonzero length")
    basis = state.basis
    coordinates = state.input_coordinates
    mixing = compression.mixing
    eigenvalues, eigenvectors = np.linalg.eigh(state.projected)
    rotated_coordinates = eigenvectors.T @ coordinates
    projected_directions = tuple(
        eigenvectors.T @ (basis.T @ family.apply_direction(j, basis)) @ eigenvectors
        for j in range(family.num_parameters)
    )
    predictions = []
    derivatives_by_time = []
    gradient = np.zeros(family.num_parameters)
    loss = 0.0
    for target, time_value in zip(target_tuple, time_tuple):
        weighted = np.cos(time_value * eigenvalues)[:, None] * rotated_coordinates
        prediction = mixing @ (rotated_coordinates.T @ weighted) @ mixing.T
        loewner = cosine_loewner(eigenvalues, time_value)
        derivatives = tuple(
            mixing @ (rotated_coordinates.T @ (loewner * direction)
                      @ rotated_coordinates) @ mixing.T
            for direction in projected_directions
        )
        error = prediction - target
        loss += 0.5 * float(np.linalg.norm(error) ** 2) / len(time_tuple)
        gradient += np.array([np.sum(error * item) for item in derivatives]) / len(time_tuple)
        predictions.append(prediction)
        derivatives_by_time.append(derivatives)
    return GradientResult(
        loss=loss,
        gradient=gradient,
        predictions=tuple(predictions),
        derivatives=tuple(derivatives_by_time),
        state=state,
        compression=compression,
    )


def evaluate_at_depth(
    family,
    theta: Array,
    probe_block: Array,
    targets: Iterable[Array],
    times: Iterable[float],
    *,
    rank: int,
    block_steps: int,
) -> GradientResult:
    compression = compress_probe(probe_block, rank)
    factorization = IncrementalBlockKrylov(family, theta, compression.coordinates)
    state = factorization.state(block_steps)
    return projected_loss_gradient(family, state, compression, targets, times)
