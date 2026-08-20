"""Each fixture exercises the defect it was written for."""

from __future__ import annotations

import pytest

from pv.policy import DEFAULT_POLICY, evidence_defects, polarity

EXPECTED_DEFECTS = [
    ("case-008", "obs-008-a", "OUT_OF_WINDOW"),
    ("case-009", "obs-009-a", "UNAPPROVED_MODEL_VERSION"),
    ("case-010", "obs-010-a", "SCOPE_MISMATCH"),
    ("case-012", "obs-012-a", "EVIDENCE_AFFECTING_TRANSFORM"),
    ("case-013", "obs-013-a", "SUPERSEDED_SOURCE"),
    ("case-014", "obs-014-a", "CAPTURE_TIME_UNKNOWN"),
    ("case-016", "obs-016-a", "LOW_CONFIDENCE"),
    ("case-017", "obs-017-a", "BROKEN_PROVENANCE"),
]


@pytest.mark.parametrize("case_id,obs_id,code", EXPECTED_DEFECTS)
def test_defect_is_detected(by_id, case_id, obs_id, code):
    graph = by_id[case_id].graph
    obs = graph.get(obs_id)
    codes = {d.code for d in evidence_defects(graph, obs, DEFAULT_POLICY)}
    assert code in codes


def test_clean_evidence_has_no_defects(by_id):
    graph = by_id["case-001"].graph
    for obs in graph.observations():
        assert evidence_defects(graph, obs, DEFAULT_POLICY) == ()


def test_benign_transformation_does_not_block_evidence(by_id):
    graph = by_id["case-002"].graph
    assert evidence_defects(graph, graph.get("obs-002-a"), DEFAULT_POLICY) == ()


def test_broken_provenance_escalates_rather_than_merely_blocking(by_id):
    graph = by_id["case-017"].graph
    defects = evidence_defects(graph, graph.get("obs-017-a"), DEFAULT_POLICY)
    assert any(d.severity == "escalate" for d in defects)


def test_off_topic_label_is_irrelevant_not_supporting(by_id):
    graph = by_id["case-003"].graph
    assert polarity(graph.claim, graph.get("obs-003-c")) == "irrelevant"
    assert polarity(graph.claim, graph.get("obs-003-a")) == "supports"
