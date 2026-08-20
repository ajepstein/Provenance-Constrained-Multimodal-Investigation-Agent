from __future__ import annotations

from pathlib import Path

import pytest

from pv.loader import load_cases

CASES_DIR = Path(__file__).resolve().parent.parent / "cases"


@pytest.fixture(scope="session")
def cases():
    return load_cases(CASES_DIR)


@pytest.fixture(scope="session")
def by_id(cases):
    return {c.case_id: c for c in cases}
