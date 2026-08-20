"""The gate: what it blocks, and what it is not allowed to do."""

from __future__ import annotations

import pytest

from pv.evaluate import corpus_index
from pv.types import Assertion, Decision, OUTCOMES
from pv.validator import case_conflicts, validate


def _decision(case, outcome, cited=(), summary="", assertions=None):
    return Decision(
        case_id=case.case_id,
        claim_id=case.graph.claim.id,
        outcome=outcome,
        assertions=tuple(assertions or [Assertion("finding", tuple(cited))])
        if cited or assertions
        else (),
        evidence_used=tuple(cited),
        summary=summary,
    )


def test_validator_never_strengthens_a_conclusion(cases):
    """Monotonicity, checked over every case and every outcome an agent could
    propose: the published answer is never less conservative than the proposal."""
    rank = {"supported": 0, "contradicted": 0, "insufficient_evidence": 1,
            "needs_human_review": 2}
    for case in cases:
        admissible = [o.id for o in case.graph.observations()]
        for outcome in OUTCOMES:
            result = validate(case.graph, _decision(case, outcome, admissible))
            assert rank[result.final_outcome] >= rank[outcome], (
                f"{case.case_id}: {outcome} -> {result.final_outcome}"
            )


def test_unresolvable_citation_is_rejected(by_id):
    case = by_id["case-001"]
    result = validate(case.graph, _decision(case, "supported", ["obs-999-z"]))
    assert "CITATION_UNRESOLVED" in result.codes()
    assert result.repairable


def test_citation_from_another_case_names_its_real_home(cases, by_id):
    case = by_id["case-001"]
    index = corpus_index(cases)
    result = validate(
        case.graph, _decision(case, "supported", ["obs-004-a"]), corpus_index=index
    )
    assert "CITATION_FOREIGN_CASE" in result.codes()
    assert any("case-004" in v.message for v in result.violations)


def test_conflict_escalates_even_when_the_agent_never_mentions_it(by_id):
    """An agent cannot dispose of a conflict by declining to cite either side."""
    case = by_id["case-011"]
    result = validate(case.graph, _decision(case, "supported", ["obs-011-b"]))
    assert result.final_outcome == "needs_human_review"
    assert "UNRESOLVED_CONFLICT" in result.codes()
    assert case_conflicts(case.graph)


def test_supersession_removes_the_apparent_conflict(by_id):
    """case-013 holds a present/absent pair, but one side rests on a
    superseded upload, so it is not an unresolved disagreement."""
    assert case_conflicts(by_id["case-013"].graph) == ()


def test_invented_quantity_in_the_summary_is_rejected(by_id):
    case = by_id["case-001"]
    result = validate(
        case.graph,
        _decision(
            case, "supported", ["obs-001-a"], summary="All 7 workers wore hard hats."
        ),
    )
    assert "UNGROUNDED_NUMBER" in result.codes()


def test_borrowed_finding_in_the_summary_is_rejected(by_id):
    case = by_id["case-003"]
    result = validate(
        case.graph,
        _decision(
            case,
            "supported",
            ["obs-003-a", "obs-003-b"],
            summary="PPE was worn; a ladder_unsecured issue was also confirmed.",
        ),
    )
    assert "UNCITED_LABEL_MENTION" in result.codes()


def test_supported_needs_every_required_item(by_id):
    case = by_id["case-001"]
    result = validate(case.graph, _decision(case, "supported", ["obs-001-a"]))
    assert "REQUIRED_LABEL_UNCOVERED" in result.codes()
    assert result.final_outcome == "insufficient_evidence"


def test_inadmissible_citation_cannot_carry_a_conclusion(by_id):
    case = by_id["case-010"]
    result = validate(
        case.graph, _decision(case, "supported", ["obs-010-a", "obs-010-b"])
    )
    assert "SCOPE_MISMATCH" in result.codes()
    assert result.final_outcome != "supported"
