from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse


EXPECTED_DIRECT = {"matplotlib": "3.9.2", "numpy": "1.26.4", "torch": "2.6.0"}
ALLIANCE_WHEELHOUSE = PurePosixPath(
    "/cvmfs/soft.computecanada.ca/custom/python/wheelhouse"
)


def audit_resolution(
    report: dict[str, Any],
    *,
    allowed_root: PurePosixPath = ALLIANCE_WHEELHOUSE,
) -> list[tuple[str, str, str]]:
    if report.get("version") != "1":
        raise ValueError(f"Unsupported pip installation report version: {report.get('version')}")
    environment = report.get("environment", {})
    expected_environment = {
        "implementation_name": "cpython",
        "platform_machine": "x86_64",
        "platform_system": "Linux",
        "python_version": "3.11",
    }
    actual_environment = {key: environment.get(key) for key in expected_environment}
    if actual_environment != expected_environment:
        raise ValueError(
            f"Narval resolution environment mismatch: expected {expected_environment}, "
            f"found {actual_environment}"
        )
    direct: dict[str, str] = {}
    resolved: list[tuple[str, str, str]] = []
    for item in report.get("install", []):
        metadata = item["metadata"]
        name = metadata["name"].lower().replace("_", "-")
        version = metadata["version"]
        url = item["download_info"]["url"]
        parsed = urlparse(url)
        path = PurePosixPath(unquote(parsed.path))
        if parsed.scheme != "file" or path.suffix.lower() != ".whl":
            raise ValueError(f"Rejected non-wheel dependency source: {url}")
        try:
            path.relative_to(allowed_root)
        except ValueError as error:
            raise ValueError(f"Rejected non-Alliance dependency source: {path}") from error
        if item.get("requested", False):
            direct[name] = version.split("+", 1)[0]
        resolved.append((name, version, str(path)))

    if direct != EXPECTED_DIRECT:
        raise ValueError(
            f"Direct dependency mismatch: expected {EXPECTED_DIRECT}, resolved {direct}"
        )
    if not resolved:
        raise ValueError("Offline dependency resolution produced an empty install plan")
    return sorted(resolved)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    resolved = audit_resolution(report)
    print("Verified offline wheel closure:")
    for name, version, path in resolved:
        print(f"  {name}=={version} <- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
