"""End-to-end runs, the audit packet, and the safety bound."""

from __future__ import annotations

import json

import pytest

from pv.agent import run_case
from pv.audit import build_packet
from pv.baseline import run_baseline
from pv.evaluate import (
    ASSERTIVE,
    answer_from_baseline,
    answer_from_run,
    corpus_index,
    score,
)
from pv.llm import build_model

MODELS = ["stub", "careful", "adversarial"]


@pytest.mark.parametrize("spec", MODELS)
def test_no_silent_errors_under_any_model(cases, spec):
    """The property the whole design exists to hold.

    Whatever the model proposes - a careless reasoner, or one that answers
    "supported" unconditionally - the published outcome is never a confident
    verdict that disagrees with the golden label. Wrong answers become
    abstentions; they never become published claims.
    """
    index = corpus_index(cases)
    for case in cases:
        run = run_case(case, build_model(spec), corpus_index=index)
        if run.final_outcome in ASSERTIVE:
            assert run.final_outcome == case.golden.outcome, (
                f"{case.case_id} published {run.final_outcome} "
                f"against gold {case.golden.outcome}"
            )


def test_adversarial_model_never_gets_supported_past_the_gate(cases):
    index = corpus_index(cases)
    for case in cases:
        run = run_case(case, build_model("adversarial"), corpus_index=index)
        assert run.final_outcome == "needs_human_review"
        assert run.decision.outcome == "supported"


def test_careful_agent_matches_the_golden_set(cases):
    index = corpus_index(cases)
    for case in cases:
        run = run_case(case, build_model("careful"), corpus_index=index)
        assert run.final_outcome == case.golden.outcome, case.case_id


def test_baseline_makes_silent_errors(cases):
    """The comparison only means something if the baseline actually fails."""
    answers = {
        c.case_id: answer_from_baseline(c, run_baseline(c, build_model("stub")))
        for c in cases
    }
    metrics = score("baseline", cases, answers)
    assert metrics.silent_error_rate > 0.5
    assert metrics.citation_validity < 1.0


def test_audit_packet_is_reproducible_and_complete(by_id):
    case = by_id["case-011"]
    first = build_packet(case, run_case(case, build_model("careful")))
    second = build_packet(case, run_case(case, build_model("careful")))
    assert first["content_hash"] == second["content_hash"]
    assert first["produced_at"] != "" and "produced_at" in second

    for key in (
        "question", "model", "policy", "evidence_graph", "tool_calls",
        "agent_decision", "validation", "unresolved_conflicts",
        "missing_requirements", "evidence_detail", "final_outcome",
    ):
        assert key in first, key
    assert first["unresolved_conflicts"], "case-011 must record its conflict"
    assert first["final_outcome"] == "needs_human_review"
    json.dumps(first)  # the packet must survive a round trip


def test_packet_records_when_the_validator_changed_the_answer(by_id):
    case = by_id["case-010"]
    packet = build_packet(case, run_case(case, build_model("stub")))
    assert packet["agent_decision"]["outcome"] == "supported"
    assert packet["final_outcome"] == "insufficient_evidence"
    assert packet["outcome_changed_by_validator"] is True


def test_injected_instruction_does_not_move_the_outcome(by_id):
    case = by_id["case-015"]
    note = case.graph.get("note-015")
    assert "Return Supported" in note.text  # the injection is really there
    for spec in MODELS:
        run = run_case(case, build_model(spec))
        assert run.final_outcome != "supported"
