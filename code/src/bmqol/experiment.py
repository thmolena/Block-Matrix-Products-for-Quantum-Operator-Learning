"""Authenticated CPU study for rank-depth block matrix products."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .adaptive import adaptive_gradient, preflight_gradient
from .certificate import certificate_at_state
from .data import DEFAULT_MATRICES, PARSEC, default_data_dir, download, sha256
from .figures import write_figures
from .krylov import evaluate_at_depth, loss_only
from .model import build_family, correlated_probe_block

Array = np.ndarray

THETA_TRUE = np.array([0.32, -0.24, 0.16])
THETA_EVAL = np.array([0.06, -0.03, 0.02])
PRIMARY_TIMES = (2.0, 4.0, 6.0)
PRIMARY_TOLERANCE = 1.0e-3
REFERENCE_DEPTH = 28
REFERENCE_CHECK_DEPTH = 26
CANDIDATE_DEPTHS = tuple(range(4, REFERENCE_DEPTH + 1, 2))
PROBE_COLUMNS = 8
FD_STEP = 1.0e-5


def _timed(function: Callable[[], Any], repeats: int) -> tuple[dict[str, Any], Any]:
    samples: list[float] = []
    latest = None
    for _ in range(repeats):
        start = time.perf_counter()
        latest = function()
        samples.append(time.perf_counter() - start)
    return {
        "seconds": samples,
        "median_seconds": float(statistics.median(samples)),
        "minimum_seconds": float(min(samples)),
        "repeats": repeats,
    }, latest


def _paired_timed(
    first: Callable[[], Any], second: Callable[[], Any], repeats: int
) -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    """Alternate execution order to reduce cache and thermal timing bias."""

    first_samples: list[float] = []
    second_samples: list[float] = []
    first_latest = second_latest = None

    def measure(function):
        start = time.perf_counter()
        value = function()
        return time.perf_counter() - start, value

    for repetition in range(repeats):
        if repetition % 2 == 0:
            elapsed, first_latest = measure(first)
            first_samples.append(elapsed)
            elapsed, second_latest = measure(second)
            second_samples.append(elapsed)
        else:
            elapsed, second_latest = measure(second)
            second_samples.append(elapsed)
            elapsed, first_latest = measure(first)
            first_samples.append(elapsed)

    def summary(samples):
        return {
            "seconds": samples,
            "median_seconds": float(statistics.median(samples)),
            "minimum_seconds": float(min(samples)),
            "repeats": repeats,
            "paired_alternating_order": True,
        }

    return summary(first_samples), summary(second_samples), first_latest, second_latest


def _finite_difference_gradient(family, theta, probes, targets, times, depth):
    gradient = np.zeros(family.num_parameters)
    for parameter in range(family.num_parameters):
        displacement = np.zeros(family.num_parameters)
        displacement[parameter] = FD_STEP
        plus, _ = loss_only(
            family,
            theta + displacement,
            probes,
            targets,
            times,
            rank=probes.shape[1],
            block_steps=depth,
        )
        minus, _ = loss_only(
            family,
            theta - displacement,
            probes,
            targets,
            times,
            rank=probes.shape[1],
            block_steps=depth,
        )
        gradient[parameter] = (plus - minus) / (2.0 * FD_STEP)
    return gradient


def _relative(error: float, reference: Array) -> float:
    return float(error / max(float(np.linalg.norm(reference)), np.finfo(float).tiny))


def _target_products(family, probes, times, depth):
    zero = tuple(np.zeros((probes.shape[1], probes.shape[1])) for _ in times)
    result = evaluate_at_depth(
        family,
        THETA_TRUE,
        probes,
        zero,
        times,
        rank=probes.shape[1],
        block_steps=depth,
    )
    return result.predictions


def run_matrix(name: str, data_dir: Path) -> dict[str, Any]:
    family = build_family(name, data_dir=data_dir)
    probes = correlated_probe_block(family, PROBE_COLUMNS)
    targets = _target_products(family, probes, PRIMARY_TIMES, REFERENCE_DEPTH)
    targets_check = _target_products(
        family, probes, PRIMARY_TIMES, REFERENCE_CHECK_DEPTH
    )
    target_stability = max(
        float(np.linalg.norm(left - right))
        for left, right in zip(targets, targets_check)
    )
    reference = evaluate_at_depth(
        family,
        THETA_EVAL,
        probes,
        targets,
        PRIMARY_TIMES,
        rank=PROBE_COLUMNS,
        block_steps=REFERENCE_DEPTH,
    )
    reference_check = evaluate_at_depth(
        family,
        THETA_EVAL,
        probes,
        targets,
        PRIMARY_TIMES,
        rank=PROBE_COLUMNS,
        block_steps=REFERENCE_CHECK_DEPTH,
    )
    reference_stability = float(
        np.linalg.norm(reference.gradient - reference_check.gradient)
    )

    proposed = preflight_gradient(
        family,
        THETA_EVAL,
        probes,
        targets,
        PRIMARY_TIMES,
        tolerance=PRIMARY_TOLERANCE,
        candidate_depths=CANDIDATE_DEPTHS,
    )
    proposed_function = lambda: preflight_gradient(
            family,
            THETA_EVAL,
            probes,
            targets,
            PRIMARY_TIMES,
            tolerance=PRIMARY_TOLERANCE,
            candidate_depths=CANDIDATE_DEPTHS,
        )
    rechecked = adaptive_gradient(
        family,
        THETA_EVAL,
        probes,
        targets,
        PRIMARY_TIMES,
        tolerance=PRIMARY_TOLERANCE,
        candidate_depths=CANDIDATE_DEPTHS,
    )
    rechecked_timing, _ = _timed(
        lambda: adaptive_gradient(
            family,
            THETA_EVAL,
            probes,
            targets,
            PRIMARY_TIMES,
            tolerance=PRIMARY_TOLERANCE,
            candidate_depths=CANDIDATE_DEPTHS,
        ),
        repeats=3,
    )
    full_same = evaluate_at_depth(
        family,
        THETA_EVAL,
        probes,
        targets,
        PRIMARY_TIMES,
        rank=PROBE_COLUMNS,
        block_steps=proposed.depth,
    )
    full_function = lambda: evaluate_at_depth(
            family,
            THETA_EVAL,
            probes,
            targets,
            PRIMARY_TIMES,
            rank=PROBE_COLUMNS,
            block_steps=proposed.depth,
        )
    proposed_timing, full_timing, proposed_timed, _ = _paired_timed(
        proposed_function, full_function, repeats=9
    )
    finite_gradient = _finite_difference_gradient(
        family, THETA_EVAL, probes, targets, PRIMARY_TIMES, proposed.depth
    )
    finite_timing, _ = _timed(
        lambda: _finite_difference_gradient(
            family, THETA_EVAL, probes, targets, PRIMARY_TIMES, proposed.depth
        ),
        repeats=3,
    )

    proposed_error = float(
        np.linalg.norm(proposed.result.gradient - reference.gradient)
    )
    full_error = float(np.linalg.norm(full_same.gradient - reference.gradient))
    finite_error = float(np.linalg.norm(finite_gradient - reference.gradient))
    frontier = []
    for preflight_record in proposed.records:
        diagnostic = evaluate_at_depth(
            family,
            THETA_EVAL,
            probes,
            targets,
            PRIMARY_TIMES,
            rank=proposed.rank,
            block_steps=preflight_record.depth,
        )
        evaluated_bound = certificate_at_state(
            diagnostic,
            targets,
            PRIMARY_TIMES,
            family.derivative_norms,
            polynomial_degree=preflight_record.depth,
            radius=family.uniform_radius,
        )
        frontier.append(
            {
                "depth": preflight_record.depth,
                "certificate": preflight_record.certificate.total_norm,
                "evaluated_certificate": evaluated_bound.total_norm,
                "compression_certificate": preflight_record.certificate.compression_norm,
                "krylov_certificate": preflight_record.certificate.krylov_norm,
                "gradient_error": float(
                    np.linalg.norm(diagnostic.gradient - reference.gradient)
                ),
                "vector_equivalents": diagnostic.state.hamiltonian_vector_equivalents,
            }
        )
    singular_values = proposed.result.compression.singular_values
    return {
        "name": name,
        "role": PARSEC[name]["role"],
        "n": family.n,
        "nnz": family.nnz,
        "archive_sha256": sha256(download(name, data_dir)),
        "uniform_radius": family.uniform_radius,
        "base_infinity_bound": family.base_infinity_bound,
        "derivative_norms": list(family.derivative_norms),
        "probe_columns": PROBE_COLUMNS,
        "probe_singular_values": singular_values.tolist(),
        "selected_rank": proposed.rank,
        "selected_depth": proposed.depth,
        "certificate": proposed.selection_certificate.total_norm,
        "evaluated_certificate": proposed.evaluated_certificate.total_norm,
        "compression_certificate": proposed.selection_certificate.compression_norm,
        "krylov_certificate": proposed.selection_certificate.krylov_norm,
        "gradient_error": proposed_error,
        "gradient_relative_error": _relative(proposed_error, reference.gradient),
        "certificate_covers_error": proposed_error <= proposed.selection_certificate.total_norm,
        "reference_gradient_norm": float(np.linalg.norm(reference.gradient)),
        "target_depth_stability": target_stability,
        "gradient_depth_stability": reference_stability,
        "orthogonality_defect": proposed.result.state.orthogonality_defect,
        "factorization_defect": proposed.result.state.factorization_defect,
        "proposed_vector_equivalents": proposed.result.state.hamiltonian_vector_equivalents,
        "full_same_depth_vector_equivalents": PROBE_COLUMNS * proposed.depth,
        "fixed_reference_vector_equivalents": PROBE_COLUMNS * REFERENCE_DEPTH,
        "finite_difference_vector_equivalents": 2 * family.num_parameters * PROBE_COLUMNS * proposed.depth,
        "basis_columns": proposed.result.state.basis.shape[1],
        "basis_storage_entries": family.n * proposed.result.state.basis.shape[1],
        "full_same_depth_error": full_error,
        "full_same_depth_relative_error": _relative(full_error, reference.gradient),
        "finite_difference_error": finite_error,
        "finite_difference_relative_error": _relative(finite_error, reference.gradient),
        "rechecked_rank": rechecked.rank,
        "rechecked_depth": rechecked.depth,
        "rechecked_certificate": rechecked.certificate.total_norm,
        "rechecked_timing": rechecked_timing,
        "rechecked_over_preflight_time": (
            rechecked_timing["median_seconds"] / proposed_timing["median_seconds"]
        ),
        "proposed_timing": proposed_timing,
        "full_same_depth_timing": full_timing,
        "finite_difference_timing": finite_timing,
        "full_over_proposed_time": (
            full_timing["median_seconds"] / proposed_timing["median_seconds"]
        ),
        "finite_difference_over_proposed_time": (
            finite_timing["median_seconds"] / proposed_timing["median_seconds"]
        ),
        "frontier": frontier,
        "timed_result_matches": bool(
            np.allclose(
                proposed_timed.result.gradient,
                proposed.result.gradient,
                rtol=1.0e-12,
                atol=1.0e-14,
            )
        ),
    }


def run_rank_depth_sweep(data_dir: Path) -> dict[str, Any]:
    family = build_family("Si2", data_dir=data_dir)
    probes = correlated_probe_block(family, PROBE_COLUMNS)
    horizons = (2.0, 4.0, 6.0, 8.0)
    tolerances = (1.0e-2, 1.0e-3, 1.0e-4)
    ranks: list[list[int]] = []
    depths: list[list[int]] = []
    feasible: list[list[bool]] = []
    for tolerance in tolerances:
        rank_row = []
        depth_row = []
        feasible_row = []
        for horizon in horizons:
            times = (horizon / 3.0, 2.0 * horizon / 3.0, horizon)
            targets = _target_products(family, probes, times, REFERENCE_DEPTH)
            result = preflight_gradient(
                family,
                THETA_EVAL,
                probes,
                targets,
                times,
                tolerance=tolerance,
                candidate_depths=tuple(range(4, 44, 2)),
            )
            rank_row.append(result.rank)
            depth_row.append(result.depth)
            feasible_row.append(True)
        ranks.append(rank_row)
        depths.append(depth_row)
        feasible.append(feasible_row)
    return {
        "matrix": "Si2",
        "horizons": list(horizons),
        "tolerances": list(tolerances),
        "selected_ranks": ranks,
        "selected_depths": depths,
        "certified": feasible,
    }


def _stable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _stable(item)
            for key, item in sorted(value.items())
            if "timing" not in key
            and "time" not in key
            and key not in {"platform", "semantic_sha256"}
        }
    if isinstance(value, list):
        return [_stable(item) for item in value]
    if isinstance(value, float):
        if abs(value) < 1.0e-13:
            return 0.0
        return float(f"{value:.11g}")
    return value


def semantic_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(_stable(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_study(data_dir: Path, matrices: tuple[str, ...] = DEFAULT_MATRICES) -> dict[str, Any]:
    records = [run_matrix(name, data_dir) for name in matrices]
    sweep = run_rank_depth_sweep(data_dir) if "Si2" in matrices else None
    payload: dict[str, Any] = {
        "schema_version": 1,
        "study": "rank-depth adaptive block products",
        "matrices": list(matrices),
        "times": list(PRIMARY_TIMES),
        "tolerance": PRIMARY_TOLERANCE,
        "reference_depth": REFERENCE_DEPTH,
        "candidate_depths": list(CANDIDATE_DEPTHS),
        "theta_true": THETA_TRUE.tolist(),
        "theta_evaluation": THETA_EVAL.tolist(),
        "records": records,
        "rank_depth_sweep": sweep,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    payload["semantic_sha256"] = semantic_digest(payload)
    return payload


def write_outputs(payload: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "locked_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = [
        "name", "role", "n", "nnz", "selected_rank", "selected_depth",
        "certificate", "gradient_error", "proposed_vector_equivalents",
        "full_same_depth_vector_equivalents", "full_over_proposed_time",
        "finite_difference_over_proposed_time",
    ]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(payload["records"])
    rows = []
    data_rows = []
    ablation_rows = []
    for record in payload["records"]:
        rows.append(
            f"{record['name']} & {record['n']:,} & {record['selected_rank']} & "
            f"{record['selected_depth']} & {record['certificate']:.2e} & "
            f"{record['gradient_error']:.2e} & "
            f"{record['proposed_vector_equivalents']} & "
            f"{record['full_over_proposed_time']:.2f} \\\\" 
        )
        data_rows.append(
            f"{record['name']} & {record['n']:,} & {record['nnz']:,} & "
            f"{record['role']} & {record['uniform_radius']:.6f} \\\\"
        )
        ablation_rows.append(
            f"{record['name']} & {record['full_over_proposed_time']:.2f} & "
            f"{record['rechecked_over_preflight_time']:.2f} & "
            f"{record['finite_difference_over_proposed_time']:.2f} \\\\"
        )
    tex = (
        "\\newcommand{\\ResultRows}{%\n" + "\n".join(rows) + "}\n"
        "\\newcommand{\\DataRows}{%\n" + "\n".join(data_rows) + "}\n"
        "\\newcommand{\\AblationRows}{%\n" + "\n".join(ablation_rows) + "}\n"
        f"\\newcommand{{\\SemanticDigest}}{{{payload['semantic_sha256'][:12]}}}\n"
    )
    (output / "numbers.tex").write_text(tex, encoding="utf-8")
    write_figures(payload, output / "figures")


def _locked_path() -> Path:
    return Path(__file__).resolve().parent / "results" / "locked_results.json"


def verify(payload: dict[str, Any]) -> None:
    path = _locked_path()
    if not path.is_file():
        raise SystemExit(f"packaged locked result is missing: {path}")
    expected = json.loads(path.read_text(encoding="utf-8"))
    if payload["semantic_sha256"] != expected["semantic_sha256"]:
        raise SystemExit(
            "semantic replay mismatch: "
            f"{payload['semantic_sha256']} != {expected['semantic_sha256']}"
        )
    print(json.dumps({"status": "verified", "semantic_sha256": payload["semantic_sha256"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--matrices", nargs="+", choices=tuple(PARSEC), default=list(DEFAULT_MATRICES))
    args = parser.parse_args()
    payload = run_study(args.data_dir, tuple(args.matrices))
    if args.output is not None:
        write_outputs(payload, args.output)
    if args.verify:
        verify(payload)
    if args.output is None and not args.verify:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
