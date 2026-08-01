"""Authenticated PARSEC Hamiltonians from the SuiteSparse Matrix Collection."""

from __future__ import annotations

import hashlib
import io
import tarfile
import urllib.request
from pathlib import Path
from typing import Any


PARSEC: dict[str, dict[str, Any]] = {
    "Si2": {
        "n": 769,
        "nnz": 17801,
        "sha256": "84e9c6f4b61a88adc1fc927e3be1fa75cb51d01f4ba26287e9c803578c0e27cd",
        "role": "development",
    },
    "SiH4": {
        "n": 5041,
        "nnz": 171903,
        "sha256": "803b52431d5e4967ac230cd5be591557d7a99c175b9ec170555151f8d76776bd",
        "role": "confirmation",
    },
    "benzene": {
        "n": 8219,
        "nnz": 242669,
        "sha256": "a1f78f39eefc4436820d9a9745e2b7c73375dce54e6bf54c6d39da0b029e89f4",
        "role": "confirmation",
    },
    "Si5H12": {
        "n": 19896,
        "nnz": 738598,
        "sha256": "895c0d8c7069e4a0401fc68838a124fbec7e4bc77f2dbd581cea8828e47578b9",
        "role": "confirmation",
    },
}

DEFAULT_MATRICES = tuple(PARSEC)
URL = "https://sparse.tamu.edu/MM/PARSEC/{name}.tar.gz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_data_dir() -> Path:
    return Path.home() / ".cache" / "bmqol" / "parsec"


def archive_path(name: str, data_dir: Path | None = None) -> Path:
    if name not in PARSEC:
        raise ValueError(f"unknown PARSEC matrix {name!r}")
    directory = Path(data_dir) if data_dir is not None else default_data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{name}.tar.gz"


def download(name: str, data_dir: Path | None = None) -> Path:
    """Return a SHA-256 authenticated archive, downloading only if necessary."""

    path = archive_path(name, data_dir)
    expected = PARSEC[name]["sha256"]
    if path.is_file() and sha256(path) == expected:
        return path
    request = urllib.request.Request(
        URL.format(name=name), headers={"User-Agent": "bmqol/1.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:  # nosec B310
        payload = response.read()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise ValueError(
            f"archive digest mismatch for {name}: {observed} != {expected}"
        )
    temporary = path.with_suffix(".tar.gz.part")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return path


def load_matrix(name: str, data_dir: Path | None = None):
    """Load one real symmetric PARSEC Hamiltonian as CSR."""

    from scipy.io import mmread

    path = download(name, data_dir)
    with tarfile.open(fileobj=io.BytesIO(path.read_bytes()), mode="r:gz") as archive:
        members = [member for member in archive.getmembers() if member.name.endswith(".mtx")]
        if len(members) != 1:
            raise ValueError(f"expected one Matrix Market file in {path}")
        stream = archive.extractfile(members[0])
        if stream is None:
            raise ValueError(f"could not read {members[0].name}")
        matrix = mmread(io.BytesIO(stream.read())).tocsr().astype(float)
    matrix = (0.5 * (matrix + matrix.T)).tocsr()
    expected = PARSEC[name]
    if matrix.shape != (expected["n"], expected["n"]):
        raise ValueError(f"unexpected shape for {name}: {matrix.shape}")
    if matrix.nnz != expected["nnz"]:
        raise ValueError(f"unexpected stored-entry count for {name}: {matrix.nnz}")
    return matrix

