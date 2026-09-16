"""Validate the packaged locked result without downloading external data."""

from __future__ import annotations

import argparse
import json
import statistics
from importlib.resources import files
from pathlib import Path

import numpy as np

from .data import PARSEC
from .experiment import semantic_digest, implementation_digest


def locked_result_path() -> Path:
    return Path(str(files("bmqol").joinpath("results/locked_results.json")))


def validate_payload(payload: dict) -> list[str]:
    errors: list[str] = []
    observed_digest = semantic_digest(payload)
    if observed_digest != payload.get("semantic_sha256"):
        errors.append("semantic digest does not match the locked payload")
    if payload.get('implementation_sha256')!=implementation_digest():
        errors.append('implementation digest does not match current executable source')
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
    if payload.get("schema_version") != 3:
        errors.append("expanded-study schema is missing")
    for record in records:
        comparisons = record.get("comparisons", {})
        frontiers = comparisons.get("frontiers", [])
        if not frontiers:
            errors.append("matched-accuracy frontiers are missing")
        for point in frontiers:
            samples = point["timing"]["seconds"]
            if len(samples) != 3 or not all(np.isfinite(x) and x > 0 for x in samples):
                errors.append("invalid frontier timing samples")
            if not np.isfinite(point["error"]) or point["error"] < 0:
                errors.append("invalid frontier discrepancy")
        for oracle in comparisons.get("observed_accuracy_oracles", []):
            feasible = [x for x in frontiers if x['method']==oracle['method']
                        and x['error']<=oracle['tolerance']]
            if not feasible or oracle['work'] != min(x['work'] for x in feasible):
                errors.append("oracle does not minimize recorded feasible work")
        for policy in comparisons.get("certified_policies", []):
            if policy['error'] > policy['certificate'] or policy['certificate'] > tolerance:
                errors.append("certified-policy error or bound exceeds its budget")
        ablations = comparisons.get('certificate_ablations', [])
        if len(ablations) != 4:
            errors.append('four certificate ablations are required')
        for case in ablations:
            if not case['error'] <= case['certificate'] <= tolerance:
                errors.append('certificate ablation failed its accuracy budget')
            component_sum=sum(case[key] for key in
                ('compression_certificate','response_certificate','derivative_certificate'))
            # Each component vector is a nonnegative scalar times the same
            # direction-norm vector, so their Euclidean norms add here.
            if not np.isclose(component_sum,case['certificate'],rtol=1e-10,atol=1e-14):
                errors.append('certificate decomposition is inconsistent')
        incremental=comparisons.get('incremental_adaptive', [])
        if len(incremental)!=6:
            errors.append('six incremental stopping comparisons are required')
        for case in incremental:
            checkpoints=case['checkpoints']
            if sum(c['incremental_work'] for c in checkpoints)!=case['work']:
                errors.append('incremental work counted more than once or omitted')
            if case['observed_tolerance_satisfied'] != (case['error']<=case['tolerance']):
                errors.append('incremental observed accuracy flag is inconsistent')
    stress = payload.get('stress', [])
    if len(stress) != 36:
        errors.append("expected 36 stress cases")
    for case in stress:
        if not np.isfinite(case['error']) or not case['error'] <= case['certificate'] <= 1e-5:
            errors.append("stress certificate failed")
    runs = payload.get('learning', {}).get('runs', [])
    if len(runs) != 36:
        errors.append("expected 36 recovery runs")
    for run in runs:
        if not run['history'] or not np.isfinite(run['parameter_error']):
            errors.append("invalid recovery trajectory")
        if run['method'] != 'fixed_depth_4':
            if any(h['certificate'] > h['tolerance'] for h in run['history']):
                errors.append("recovery plan exceeds its stage budget")
    larger=payload.get('scalable_learning', {})
    config=larger.get('config', {})
    larger_runs=larger.get('runs', [])
    if len(larger_runs)!=288:
        errors.append('expected 288 larger learning runs')
    identifiers=set()
    for run in larger_runs:
        identity=tuple(run[key] for key in ('n','truth_index','start_index','noise',
                         'noise_seed','stationarity_tolerance','method'))
        if identity in identifiers:
            errors.append('duplicate larger-learning case')
        identifiers.add(identity)
        if run['n'] not in config['sizes'] or run['method'] not in config['methods']:
            errors.append('larger-learning case outside registered configuration')
        if not np.all(np.isfinite(run['estimate'])) or np.max(np.abs(run['estimate']))>.5+1e-12:
            errors.append('nonfinite or infeasible recovered parameters')
        if run['optimization_reference_work']>run['reference_work']:
            errors.append('optimization reference work exceeds total reference work')
        if run['optimization_seconds']>run['seconds']:
            errors.append('optimization time exceeds audited total time')
        if run['status']=='projected_stationarity' and run['projected_gradient_norm']>run['stationarity_tolerance']+1e-10:
            errors.append('independent projected stationarity check failed')
        evaluations=run['evaluations']
        if not evaluations:
            errors.append('missing learning gradient evaluations')
        if run['method']!='reference' and sum(e['basis_dimension'] for e in evaluations)!=run['work']:
            errors.append('larger-learning Krylov action ledger inconsistent')
        for evaluation in evaluations:
            bound=evaluation['gradient_bound']
            if bound is not None and evaluation['independent_gradient_error']>bound+1e-12:
                errors.append('evaluated learning certificate exceeds numerical audit allowance')
            if not 0<=evaluation['basis_dimension']<=run['n']:
                errors.append('invalid learning basis dimension')
        for step in run['history']:
            if step['directional_derivative_upper']>=0:
                errors.append('accepted direction lacks a negative directional bound')
            if step['independent_directional_derivative']>=1e-12:
                errors.append('accepted direction fails independent descent audit')
            evaluation=evaluations[step['evaluation_index']]
            if step['evaluation_theta']!=evaluation['theta']:
                errors.append('accepted step uses a different gradient evaluation point')
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
