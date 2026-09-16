# Certified Rank–Depth Selection for Block Matrix-Function Gradients

This repository contains the manuscript, implementation, and reproducible experiments for selecting probe rank and block-Krylov depth with an a priori inexact-gradient bound. The joint search minimizes nominal operator-vector work over a finite certified grid. It does not minimize runtime or certify the entire floating-point computation.

## Main findings

- On four authenticated PARSEC Hamiltonians (dimensions 769–19,896), loss-aware compression and Chebyshev derivative tails select rank 3 and depth 16: 48 operator-vector applications versus 120 for the original linear/Taylor joint rule and 128 for refined full-rank certification.
- The four-way ablation separates compression savings from depth savings. Independent gradient discrepancies remain below `9e-9` at the requested `1e-3` tolerance.
- Incremental full-rank adjoint stopping reuses its basis and meets the tested observed tolerances. At `1e-3`, it is faster than joint certification despite using 64 rather than 48 operator-vector applications. Post hoc accuracy oracles are reported separately.
- A 288-run larger learning study crosses chain sizes 256/1024, three truths, two starts, clean data plus two noisy realizations, two stationarity targets, and four methods. All recover within `1e-3`; compressed gradients support independently checked feasible descent. Strict final accuracy can require full probe rank and erase work savings. The fixed-depth control remains faster.
- The original 36 recovery runs remain as a diagnostic control: every method recovers three of six cases per noise level. Independent projected gradients and Hessians identify constrained stationary fits and an interior nonglobal minimum.
- All 36 stress cases pass the independent-reference audit; 12 now compress probe rank and 17 reach the full matrix dimension.

## Files

- [main.tex](main.tex) and [main.pdf](main.pdf): theory, protocols, results, and limitations.
- [code/](code/): installable implementation and tests.
- [locked_results.json](code/results/locked_results.json): independent-reference comparisons, timing samples, matched-accuracy frontiers, recovery trajectories, and stress cases.
- [index.html](index.html): project overview.

## Reproduce

Python 3.10–3.13 is supported. The current experiment used Python 3.12.13, NumPy 2.5.3, SciPy 1.18.1, and Matplotlib 3.11.2. Numerical and timing results may vary with the platform; versions and backend information are recorded.

```bash
cd code
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
VECLIB_MAXIMUM_THREADS=1 pytest -W error
VECLIB_MAXIMUM_THREADS=1 bmqol-reproduce --output results
bmqol-validate results/locked_results.json
```

The downloader verifies pinned SHA-256 digests and caches external archives outside the repository. The complete replay overwrites the existing results and five figure pairs. To check numerical reproducibility without overwriting the locked reference, run `bmqol-reproduce --verify` using the same numerical runtime. If `--verify` and `--output` are combined, verification occurs before writing. No reference result is used by the certified stopping rule; reference-based oracle choices are explicitly post hoc. The scalar-polarization and efficient projected-adjoint comparators are our implementations, not reproductions of another author's software. The result records include both a timing-independent numerical digest and a SHA-256 digest of the exact executable source and dependency specification. `main.tex` includes the generated `code/results/numbers.tex`; this dependency is part of the repository.

## Scientific scope

The method is classical numerical linear algebra specialized to cosine responses. The feasible-direction test uses an exact-arithmetic gradient bound; line-search acceptance uses independent floating-point exponential actions with their costs counted. This is not end-to-end interval-certified optimization. Basis storage is recorded, not peak process memory. PARSEC probes and parameter directions are constructed numerical inputs; the lattice experiment uses simulated real transition amplitudes with additive Gaussian noise, not device or finite-shot observations. Certification does not establish identifiability, global optimizer convergence, quantum advantage, or universal runtime improvement.

## License

Code and repository text use the MIT License. External matrices remain subject to their source terms.
