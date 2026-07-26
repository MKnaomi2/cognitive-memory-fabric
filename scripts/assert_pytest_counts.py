"""Fail CI unless the neural test job reports exactly 48 passes and no skips."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--expected", type=int, default=48)
    args = parser.parse_args()
    root = ET.parse(args.report).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        raise SystemExit("pytest report has no testsuite")
    counts = {
        name: int(suite.attrib.get(name, "0"))
        for name in ("tests", "failures", "errors", "skipped")
    }
    passed = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    if (
        passed != args.expected
        or counts["failures"]
        or counts["errors"]
        or counts["skipped"]
    ):
        raise SystemExit(f"pytest count gate failed: passed={passed}, {counts}")
    print(f"pytest count gate: {passed} passed, 0 skipped")


if __name__ == "__main__":
    main()
