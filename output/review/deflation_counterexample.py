"""Read-only reproducer for the manuscript review of September 15, 2026.

Run from the repository root with a supported Python and installed dependencies:
    PYTHONPATH=code/src python output/review/deflation_counterexample.py

This prints evidence; it does not change implementation or result files.
"""
import json
import platform
import numpy as np
import scipy
from scipy.linalg import expm_frechet
from scipy.sparse import csr_matrix
from bmqol.adaptive import preflight_gradient
from bmqol.krylov import IncrementalBlockKrylov, _orthogonal_block
from bmqol.model import HamiltonianFamily

E = np.eye(6)
B = E[:, 3:6]
X = E[:, [0, 0, 1]]
A = X @ B.T + B @ X.T
family = HamiltonianFamily(csr_matrix(A), (csr_matrix(A),),
                           "exact_partial_deflation", 2.0, 0.5)
Q = _orthogonal_block(X, [B], 1e-12)
state = IncrementalBlockKrylov(family, np.zeros(1), B).state(2)
C = state.basis.T @ B
result = preflight_gradient(family, np.zeros(1), B, [np.zeros((3, 3))],
                            [1.0], tolerance=1e-8,
                            candidate_depths=(20, 24, 28), selection="full")
F, L = expm_frechet(-1j * A, -1j * A)
reference_gradient = float(np.sum((B.T @ F.real @ B) * (B.T @ L.real @ B)))
analytic_gradient = float(-np.sqrt(2) * np.sin(np.sqrt(2)) * np.cos(np.sqrt(2))
                          - np.sin(1.0) * np.cos(1.0))
evidence = {
    "python": platform.python_version(),
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "candidate_rank": int(np.linalg.matrix_rank(X)),
    "returned_block_width": Q.shape[1],
    "discarded_span_residual": float(np.linalg.norm(X - Q @ (Q.T @ X))),
    "exact_K2_dimension": int(np.linalg.matrix_rank(np.column_stack((B, A @ B)))),
    "returned_K2_dimension": state.basis.shape[1],
    "orthogonality_defect": state.orthogonality_defect,
    "quadratic_moment_error": float(np.linalg.norm(
        B.T @ A @ A @ B - C.T @ state.projected @ state.projected @ C)),
    "requested_tolerance": 1e-8,
    "selection_bound": result.selection_certificate.total_norm,
    "evaluated_bound": result.evaluated_certificate.total_norm,
    "returned_depth": result.depth,
    "computed_gradient": float(result.result.gradient[0]),
    "reference_gradient": reference_gradient,
    "analytic_gradient": analytic_gradient,
    "reference_vs_analytic": abs(reference_gradient - analytic_gradient),
    "gradient_discrepancy": float(abs(result.result.gradient[0] - reference_gradient)),
}
print(json.dumps(evidence, indent=2))
