"""Uniformly bounded affine Hamiltonian families and correlated probe blocks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import load_matrix

Array = np.ndarray


@dataclass(frozen=True)
class HamiltonianFamily:
    """Affine real-symmetric family with a proved parameter-box norm bound."""

    base: object
    fields: tuple[Array, ...]
    name: str
    base_infinity_bound: float
    parameter_radius: float

    @property
    def n(self) -> int:
        return int(self.base.shape[0])

    @property
    def nnz(self) -> int:
        return int(self.base.nnz)

    @property
    def num_parameters(self) -> int:
        return len(self.fields)

    @property
    def derivative_norms(self) -> tuple[float, ...]:
        return tuple(float(np.max(np.abs(field))) for field in self.fields)

    @property
    def uniform_radius(self) -> float:
        return float(
            self.base_infinity_bound
            + self.parameter_radius * sum(self.derivative_norms)
        )

    def validate_theta(self, theta: Array) -> Array:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.num_parameters,):
            raise ValueError("theta has the wrong shape")
        if np.max(np.abs(theta)) > self.parameter_radius + 1.0e-14:
            raise ValueError("theta leaves the certified parameter box")
        return theta

    def apply(self, theta: Array, block: Array) -> Array:
        theta = self.validate_theta(theta)
        block = np.asarray(block, dtype=float)
        result = self.base @ block
        for coefficient, field in zip(theta, self.fields):
            result = result + coefficient * field[:, None] * block
        return np.asarray(result)

    def dense(self, theta: Array) -> Array:
        theta = self.validate_theta(theta)
        matrix = self.base.toarray()
        diagonal = sum(
            (coefficient * field for coefficient, field in zip(theta, self.fields)),
            start=np.zeros(self.n),
        )
        matrix[np.diag_indices(self.n)] += diagonal
        return matrix


def build_family(
    name: str,
    data_dir: Path | None = None,
    *,
    num_parameters: int = 3,
    base_budget: float = 0.80,
    field_norm: float = 0.12,
    parameter_radius: float = 0.50,
) -> HamiltonianFamily:
    """Construct a family satisfying ``||A(theta)||_2 <= 0.98`` on its box.

    The proof uses ``||H||_2 <= ||H||_infinity`` for a symmetric matrix and
    the exact infinity norm of every diagonal direction.  No iterative
    eigenvalue estimate enters the enclosure.
    """

    if not 1 <= num_parameters <= 6:
        raise ValueError("num_parameters must lie in [1, 6]")
    raw = load_matrix(name, data_dir)
    row_sum = float(np.asarray(np.abs(raw).sum(axis=1)).max())
    if not np.isfinite(row_sum) or row_sum <= 0.0:
        raise ValueError("invalid row-sum norm")
    base = (base_budget / row_sum) * raw
    grid = np.linspace(0.0, 1.0, raw.shape[0], endpoint=False)
    centers = np.linspace(0.18, 0.82, num_parameters)
    width = 0.085
    fields = tuple(
        field_norm * np.exp(-0.5 * ((grid - center) / width) ** 2)
        for center in centers
    )
    family = HamiltonianFamily(
        base=base.tocsr(),
        fields=fields,
        name=name,
        base_infinity_bound=base_budget,
        parameter_radius=parameter_radius,
    )
    if family.uniform_radius > 1.0:
        raise ValueError("construction does not fit inside the unit spectral box")
    return family


def correlated_probe_block(family: HamiltonianFamily, columns: int = 8) -> Array:
    """Return a deterministic correlated numerical excitation block.

    The block is constructed from overlapping Gaussian index envelopes centered
    near the largest diagonal magnitude of the authenticated Hamiltonian.  The
    construction is a numerical probe design, not an experimental observable or
    an atomic-coordinate model.
    """

    if not 2 <= columns <= 16:
        raise ValueError("columns must lie in [2, 16]")
    diagonal = np.asarray(family.base.diagonal())
    center = (int(np.argmax(np.abs(diagonal))) + 0.5) / family.n
    grid = np.linspace(0.0, 1.0, family.n, endpoint=False)
    offsets = np.linspace(-0.035, 0.035, columns)
    width = 0.12
    block = np.column_stack(
        [np.exp(-0.5 * ((grid - (center + offset)) / width) ** 2) for offset in offsets]
    )
    phase = np.cos(2.0 * np.pi * grid)[:, None]
    block = block * (1.0 + 0.08 * phase * np.linspace(-1.0, 1.0, columns)[None, :])
    norms = np.linalg.norm(block, axis=0)
    return block / norms[None, :]

