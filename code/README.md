# Reproduction package

The package computes block cosine responses and inexact loss gradients, with joint rank–depth selection under an exact-arithmetic error bound. Both diagonal and symmetric sparse parameter directions are supported. Reduced eigendecompositions are reused and cosine divided differences use a stable sine–sinc identity.

## Installation and complete replay

Use Python 3.10–3.13:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
VECLIB_MAXIMUM_THREADS=1 pytest -W error
VECLIB_MAXIMUM_THREADS=1 bmqol-reproduce --output results
bmqol-validate results/locked_results.json
```

The replay downloads and authenticates four external PARSEC archives, executes comparisons and recovery experiments, and overwrites `results/locked_results.json`, `results/summary.csv`, `results/numbers.tex`, and the five existing figure pairs. Timings request one BLAS thread and retain all repetition samples. The current replay used Python 3.12.13, NumPy 2.5.3, SciPy 1.18.1, and Matplotlib 3.11.2.

Use `--data-dir /path/to/cache` to reuse an authenticated archive cache. `--verify` compares the rounded, timing-independent numerical digest with the packaged result before any requested output is written; differences across numerical libraries require investigation and are not automatically evidence of a scientific failure. Timings are excluded, but physical times and all error-frontier settings are included in the digest.

## Experiments

1. Four PARSEC benchmarks: independent sparse augmented-exponential references; full and compressed depth sweeps; scalar polarization and efficient projected-adjoint comparators; joint/full/rank-first policies; basis-reusing adaptive stopping at `1e-3`, `1e-6`, `1e-9`; central differences.
2. Certificate ablation: cross `compression_method="legacy"`/`"loss-aware"` and `tail_method="taylor"`/`"chebyshev"`. Defaults use both refinements. Bound records separate compression, response, and derivative contributions. The new combined rule uses 48 operator-vector applications on all four matrices, versus 120 under the original rule.
3. Matched observed accuracy: post hoc minimum-work choices on recorded grids. These require a reference and are not executable stopping policies. The incremental heuristic is separately executable; its compressed variant freezes the primary-tolerance rank and can miss tighter accuracy targets through compression bias.
4. Original 36 tight-binding recovery runs: retain legacy/Taylor selection and the original optimizer, including failed fits. Independent projected-gradient and interior Hessian diagnostics distinguish poor local solutions from optimizer nonconvergence.
5. Larger learning: 288 runs over sizes 256/1024, three truths, two starts, noiseless data plus two independent amplitude-noise realizations, stationarity targets `1e-5`/`1e-7`, and joint/full/fixed-depth/reference-gradient methods. All use projected damped Gauss–Newton and independent loss acceptance. Certified methods tighten the gradient budget until the feasible direction is certified. Every evaluated gradient and accepted direction is checked independently afterward.
6. Thirty-six stress cases: widths 4/8/16, one/six off-diagonal directions, singular spectra spanning 0/1/3 decades, and two time horizons.

Learning records separate optimization time and operator work from post hoc validation costs. Exponential-action ledgers include norm-estimation and transpose actions. Accepted iterates and their gradient evaluation points are stored separately. Successful recovery on this pilot-informed simulated measurement design is not a global identifiability result. Strict tolerances can remove compression savings, and fixed-depth learning is faster in the reported study.

## Validation

Tests compare divided differences and projected derivatives with a separate dense exponential Fréchet implementation, explicitly check the two polynomial-exactness degrees, and enumerate the rank–depth grid independently. The result validator checks timing samples, oracle minima, all certificate ablations, cumulative incremental work, full learning case coverage, independently checked stationarity and descent, action accounting, and implementation and numerical digests. From the repository root:

```bash
python code/scripts/validate_release.py
```

The existing package-data result files under `src/bmqol/results/` are synchronized on complete output generation. Source distributions include the package data; matrix archives are not redistributed. A basis-storage count is not a peak process-memory measurement. Tail rounding does not provide end-to-end interval certification.
