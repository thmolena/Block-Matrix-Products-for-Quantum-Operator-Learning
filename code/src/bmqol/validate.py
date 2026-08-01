"""Validate the packaged locked result without downloading external data."""

from __future__ import annotations

import argparse
import json
import statistics
from importlib.resources import files
from pathlib import Path

import numpy as np

from .data import PARSEC
from .experiment import semantic_digest


def locked_result_path() -> Path:
    return Path(str(files("bmqol").joinpath("results/locked_results.json")))


def validate_payload(payload: dict) -> list[str]:
    errors: list[str] = []
    observed_digest = semantic_digest(payload)
    if observed_digest != payload.get("semantic_sha256"):
        errors.append("semantic digest does not match the locked payload")
    tolerance = float(payload.get("tolerance", np.nan))
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        errors.append("invalid registered tolerance")
    records = payload.get("records", [])
    if [record.get("name") for record in records] != payload.get("matrices"):
        errors.append("record order does not match the matrix registry")
    for record in records:
        name = record.get("name", "<missing>")
        registry = PARSEC.get(name)
        if registry is None:
            errors.append(f"{name}: absent from the authenticated registry")
            continue
        for key in ("n", "nnz", "role"):
            if record.get(key) != registry[key]:
                errors.append(f"{name}: {key} disagrees with the registry")
        if record.get("archive_sha256") != registry["sha256"]:
            errors.append(f"{name}: archive digest disagrees with the registry")
        rank = int(record["selected_rank"])
        depth = int(record["selected_depth"])
        if record["proposed_vector_equivalents"] != rank * depth:
            errors.append(f"{name}: proposed action ledger is inconsistent")
        if record["full_same_depth_vector_equivalents"] != 8 * depth:
            errors.append(f"{name}: full-block action ledger is inconsistent")
        if record["finite_difference_vector_equivalents"] != 2 * 3 * 8 * depth:
            errors.append(f"{name}: finite-difference action ledger is inconsistent")
        if record["gradient_error"] > record["certificate"]:
            errors.append(f"{name}: certificate does not cover observed error")
        if record["certificate"] > tolerance:
            errors.append(f"{name}: selected certificate exceeds tolerance")
        if not record.get("certificate_covers_error", False):
            errors.append(f"{name}: stored coverage flag is false")
        if not record.get("timed_result_matches", False):
            errors.append(f"{name}: timed result differs from the registered result")
        for label in ("proposed_timing", "full_same_depth_timing"):
            timing = record[label]
            samples = timing.get("seconds", [])
            if len(samples) != 9 or timing.get("repeats") != 9:
                errors.append(f"{name}: {label} does not contain nine repetitions")
            elif not np.isclose(statistics.median(samples), timing["median_seconds"]):
                errors.append(f"{name}: {label} median is inconsistent")
    sweep = payload.get("rank_depth_sweep")
    if sweep is None or not all(all(row) for row in sweep.get("certified", [])):
        errors.append("rank-depth sweep is absent or contains an uncertified cell")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=locked_result_path())
    args = parser.parse_args()
    payload = json.loads(args.path.read_text(encoding="utf-8"))
    errors = validate_payload(payload)
    if errors:
        raise SystemExit("validation failed:\n- " + "\n- ".join(errors))
    print(
        json.dumps(
            {
                "status": "valid",
                "records": len(payload["records"]),
                "semantic_sha256": payload["semantic_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
