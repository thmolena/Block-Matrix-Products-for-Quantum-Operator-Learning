# Rank--Depth Adaptive Block Matrix Products for Quantum Operator Learning

This repository contains a new manuscript and complete CPU reproduction package for certified block matrix-function gradients. The method chooses both the numerical rank of a correlated probe block and the block-Krylov depth before the first Hamiltonian application. Its certificate combines a parameter-box spectral enclosure, an explicit singular-value truncation bound, polynomial exactness, and positive Taylor majorants for the cosine response and its Fréchet derivative.

## Main result

At gradient tolerance `1e-3`, the locked study selects rank five or six from eight probes and depth 24 on four checksum-authenticated PARSEC Hamiltonians. The deterministic Hamiltonian-vector equivalent count falls from 192 to 120--144. Nine paired CPU repetitions favor the proposed method on Si2, SiH4, and benzene by factors 1.36, 1.65, and 2.04. The largest case, Si5H12, is retained as a negative timing result: the uncompressed block is approximately one percent faster.

Every numerical quantity in the paper is generated from `code/results/locked_results.json`. The authentic inputs are distinguished from the constructed diagonal fields, numerical probes, and targets throughout the manuscript.

## Repository contents

- [`main.tex`](main.tex): complete source with proofs, limitations, and data protocol.
- [`main.pdf`](main.pdf): compiled manuscript.
- [`code/`](code/): installable implementation, tests, authenticated downloader, locked outputs, and figure generator.
- [`index.html`](index.html): compact project page presenting the method and primary evidence.
- [`code/results/locked_results.json`](code/results/locked_results.json): full numerical record and all timing samples.

## Reproduction

```bash
cd code
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
pytest
bmqol-validate results/locked_results.json
bmqol-reproduce --output results
```

The full replay downloads approximately 430 MB of compressed SuiteSparse archives and verifies each SHA-256 digest before parsing. Runtime values depend on the CPU and sparse kernel; the timing-independent semantic digest verifies the numerical replay.

## Scientific scope

The work establishes a deterministic gradient-error certificate and a work reduction under correlated probes. It does not establish quantum advantage, universal wall-time acceleration, recovery from device measurements, or chemical inference from the PARSEC inputs. These boundaries and the largest-matrix timing exception are part of the reported result.

## License

The source code and repository text are released under the MIT License. External matrices remain subject to the terms of the SuiteSparse Matrix Collection.
