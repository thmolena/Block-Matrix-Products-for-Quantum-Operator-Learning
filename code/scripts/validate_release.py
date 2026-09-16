"""Check the manuscript/reproduction closure from the repository root."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE / "src"))

from bmqol.validate import validate_payload  # noqa: E402


def main() -> None:
    required = [
        ROOT / "main.tex",
        ROOT / "main.pdf",
        ROOT / "README.md",
        ROOT / "index.html",
        CODE / "README.md",
        CODE / "pyproject.toml",
        CODE / "results" / "locked_results.json",
        CODE / "results" / "summary.csv",
        CODE / "results" / "numbers.tex",
    ]
    errors = [f"missing required artifact: {path.relative_to(ROOT)}" for path in required if not path.is_file()]
    tex = (ROOT / "main.tex").read_text(encoding="utf-8")
    for relative in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex):
        if not (ROOT / relative).is_file():
            errors.append(f"missing manuscript figure: {relative}")
    for path in (ROOT / "main.tex", ROOT / "README.md", ROOT / "index.html"):
        text = path.read_text(encoding="utf-8")
        if re.search(r"\b(?:ChatGPT|agentic|artificial intelligence|AI)\b", text, re.IGNORECASE):
            errors.append(f"prohibited wording appears in {path.name}")
    payload = json.loads((CODE / "results" / "locked_results.json").read_text(encoding="utf-8"))
    errors.extend(validate_payload(payload))
    for filename in ("locked_results.json", "numbers.tex", "summary.csv"):
        if (CODE / "results" / filename).read_bytes() != (CODE / "src" / "bmqol" / "results" / filename).read_bytes():
            errors.append(f"packaged result differs: {filename}")
    if tex.count("\\begin{figure}") != 5:
        errors.append("expected five generated figures")
    if errors:
        raise SystemExit("release validation failed:\n- " + "\n- ".join(errors))
    print(json.dumps({"status": "release-valid", "semantic_sha256": payload["semantic_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
