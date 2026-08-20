"""The golden set has to be trustworthy before anything measured against it is."""

from __future__ import annotations

from pv.loader import CaseLoadError, load_case
from pv.policy import admissible_observations, polarity
from pv.types import OUTCOMES
from pv.validator import case_conflicts, case_missing_requirements

import pytest


def test_every_case_loads_and_is_labelled(cases):
    assert 10 <= len(cases) <= 20
    for case in cases:
        assert case.golden.outcome in OUTCOMES
        assert case.golden.rationale.strip(), f"{case.case_id} has no rationale"


def test_golden_set_covers_every_required_category(cases):
    tags = {tag for case in cases for tag in case.tags}
    for required in ("supported", "contradicted", "insufficient", "adversarial"):
        assert required in tags
    outcomes = {c.golden.outcome for c in cases}
    assert outcomes == set(OUTCOMES)


def test_golden_missing_requirements_match_the_evidence(cases):
    for case in cases:
        assert set(case_missing_requirements(case.graph)) == set(case.golden.missing), (
            case.case_id
        )


def test_golden_conflicts_match_the_graph(cases):
    for case in cases:
        found = {c.label for c in case_conflicts(case.graph)}
        assert found == set(case.golden.conflicts), case.case_id


def test_golden_evidence_is_admissible_and_relevant(cases):
    """A label that cites inadmissible or off-topic evidence would make the
    citation metrics meaningless."""
    for case in cases:
        admissible = {o.id for o in admissible_observations(case.graph)}
        for obs_id in case.golden.evidence:
            assert obs_id in admissible, f"{case.case_id}: {obs_id} is not admissible"
            obs = case.graph.get(obs_id)
            assert polarity(case.graph.claim, obs) != "irrelevant", (
                f"{case.case_id}: {obs_id} does not bear on the claim"
            )


def test_loader_rejects_a_malformed_fixture(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("case_id: x\nclaim: {id: c, text: t}\nexpected: {outcome: supported}\n")
    with pytest.raises(CaseLoadError):
        load_case(bad)
