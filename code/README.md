# Reproduction package

This directory implements the rank--depth adaptive block-product method in the accompanying manuscript. The default study downloads four PARSEC matrices from the SuiteSparse Matrix Collection, verifies their pinned SHA-256 digests, constructs the registered numerical learning problem, executes every baseline, and writes the complete numerical record and all figures.

## Installation

Python 3.10--3.13 is supported. A clean local installation is:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
pytest
bmqol-validate results/locked_results.json
```

## Full replay

The complete CPU study requires approximately 430 MB of external compressed matrix archives. Downloads are authenticated before parsing and remain in `~/.cache/bmqol/parsec` by default.

```bash
bmqol-reproduce --output results
bmqol-validate results/locked_results.json
```

To reuse a different authenticated cache:

```bash
bmqol-reproduce --data-dir /path/to/parsec-cache --output results
```

Runtime values are expected to vary across machines. Verification compares the timing-independent semantic digest and checks the recorded numerical invariants. `results/locked_results.json` retains every timing repetition rather than only summary statistics.

## Directory contract

- `src/bmqol/`: implementation, certificates, authenticated loader, experiment, plots, and validator.
- `tests/`: dense small-matrix theorem checks and cached-archive checks.
- `data/checksums.sha256`: archive authentication registry.
- `results/locked_results.json`: complete locked result, including negative timing evidence.
- `results/summary.csv`: concise table source.
- `results/numbers.tex`: generated manuscript rows.
- `results/figures/`: PDF publication figures and PNG inspection copies.

The PARSEC matrices are authentic external inputs. The diagonal parameter fields, correlated numerical probes, and targets are constructed deterministically; they are not presented as experimental observations or chemical models.
