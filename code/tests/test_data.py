from __future__ import annotations

from pathlib import Path

import pytest

from bmqol.data import PARSEC, default_data_dir, download, sha256


@pytest.mark.parametrize("name", tuple(PARSEC))
def test_registered_archives_when_cached(name: str) -> None:
    caches = (default_data_dir(), Path.home() / ".cache" / "ccbkqol" / "parsec")
    candidates = (cache / f"{name}.tar.gz" for cache in caches)
    cached = next((path for path in candidates if path.is_file()), None)
    if cached is None:
        pytest.skip("authenticated archive is not cached")
    assert sha256(cached) == PARSEC[name]["sha256"]
    assert download(name, cached.parent) == cached
