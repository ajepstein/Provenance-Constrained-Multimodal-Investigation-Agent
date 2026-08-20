"""Scoring the two systems against the hand-labelled golden set.

Accuracy alone would reward a system that always abstains, so the metric set
is deliberately two-sided: it measures how often a system is *wrong while
confident* (the failure that matters here) and how often it abstains on a case
it should have decided (the cost of the constraint).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from pv.agent import RunRecord
from pv.baseline import BaselineResult
from pv.loader import Case
from pv.policy import DEFAULT_POLICY, Policy, evidence_defects, polarity
from pv.types import Assertion, Decision
from pv.validator import case_conflicts, validate

ASSERTIVE = ("supported", "contradicted")
ABSTAINING = ("insufficient_evidence", "needs_human_review")

# An answer carries an unsupported claim when it asserts more than its
# citations can bear: it names evidence that does not resolve, leans on
# evidence that was not admitted, concludes in a direction its own citations
# contradict, or states a quantity that appears nowhere in the evidence.
UNSUPPORTED_CLAIM_CODES = (
    "UNSUPPORTED_ASSERTION",
    "UNGROUNDED_NUMBER",
    "UNCITED_LABEL_MENTION",
    "CITATION_UNRESOLVED",
    "CITATION_FOREIGN_CASE",
    "CITATION_NOT_EVIDENCE",
    "REQUIRED_LABEL_UNCOVERED",
    "OUTCOME_AGAINST_EVIDENCE",
)


@dataclass
class Answer:
    """One system's answer to one case, in a form both paths can produce."""

    case_id: str
    outcome: str
    cited: tuple[str, ...] = ()
    summary: str = ""
    reported_conflicts: tuple[str, ...] = ()
    reported_missing: tuple[str, ...] = ()
    unsupported_claim: bool = False


def answer_from_run(run: RunRecord) -> Answer:
    decision = run.decision
    validation = run.validation
    return Answer(
        case_id=run.case_id,
        outcome=run.final_outcome,
        cited=tuple(decision.evidence_used) if decision else (),
        summary=decision.summary if decision else "",
        reported_conflicts=tuple(
            sorted({c.label for c in (validation.conflicts if validation else ())})
        ),
        reported_missing=tuple(validation.missing) if validation else (),
        # The published answer cannot carry an unsupported claim: a decision
        # that fails the grounding check is rejected rather than accepted.
        unsupported_claim=False,
    )


def answer_from_baseline(
    case: Case, result: BaselineResult, policy: Policy = DEFAULT_POLICY
) -> Answer:
    """Score the baseline by running the validator over it in shadow mode.

    The validator does not gate the baseline - that is the point of the
    comparison - but running it lets both systems be measured on the same
    definition of an unsupported claim.
    """
    shadow = Decision(
        case_id=case.case_id,
        claim_id=case.graph.claim.id,
        outcome=result.outcome,  # type: ignore[arg-type]
        assertions=(Assertion(statement=result.summary, cited=result.cited),)
        if result.cited
        else (),
        evidence_used=result.cited,
        summary=result.summary,
    )
    verdict = validate(case.graph, shadow, policy)
    unsupported = any(v.code in UNSUPPORTED_CLAIM_CODES for v in verdict.violations)
    return Answer(
        case_id=case.case_id,
        outcome=result.outcome,
        cited=result.cited,
        summary=result.summary,
        reported_conflicts=(),
        reported_missing=(),
        unsupported_claim=unsupported,
    )


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


@dataclass
class Metrics:
    system: str
    n: int
    outcome_accuracy: float
    silent_error_rate: float
    citation_validity: float
    citation_recall: float
    correct_abstention: float
    over_abstention: float
    conflict_recall: float
    unsupported_claim_rate: float
    silent_errors: list[str] = field(default_factory=list)
    wrong: list[str] = field(default_factory=list)

    def row(self) -> str:
        return (
            f"{self.system:<22} {self.outcome_accuracy:>8.2f} "
            f"{self.silent_error_rate:>9.2f} {self.citation_validity:>10.2f} "
            f"{self.citation_recall:>9.2f} {self.correct_abstention:>10.2f} "
            f"{self.over_abstention:>10.2f} {self.conflict_recall:>9.2f} "
            f"{self.unsupported_claim_rate:>12.2f}"
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "cases": self.n,
            "outcome_accuracy": round(self.outcome_accuracy, 4),
            "silent_error_rate": round(self.silent_error_rate, 4),
            "citation_validity": round(self.citation_validity, 4),
            "citation_recall": round(self.citation_recall, 4),
            "correct_abstention": round(self.correct_abstention, 4),
            "over_abstention": round(self.over_abstention, 4),
            "conflict_recall": round(self.conflict_recall, 4),
            "unsupported_claim_rate": round(self.unsupported_claim_rate, 4),
            "silent_errors": self.silent_errors,
            "wrong_outcomes": self.wrong,
        }


HEADER = (
    f"{'system':<22} {'accuracy':>8} {'silent':>9} {'cite-valid':>10} "
    f"{'cite-rec':>9} {'abstain-ok':>10} {'over-abst':>10} {'conflict':>9} "
    f"{'unsupported':>12}"
)


def score(
    system: str,
    cases: Iterable[Case],
    answers: dict[str, Answer],
    policy: Policy = DEFAULT_POLICY,
) -> Metrics:
    cases = list(cases)
    n = len(cases)
    correct = 0
    silent_errors: list[str] = []
    wrong: list[str] = []
    cited_total = 0
    cited_valid = 0
    recall_scores: list[float] = []
    abstain_cases = 0
    abstain_ok = 0
    decide_cases = 0
    over_abstained = 0
    conflict_cases = 0
    conflict_found = 0
    unsupported = 0

    for case in cases:
        answer = answers[case.case_id]
        gold = case.golden

        if answer.outcome == gold.outcome:
            correct += 1
        else:
            wrong.append(case.case_id)
            # A wrong answer that is *confident* is the failure that matters:
            # the system published a verdict nobody was warned about.
            if answer.outcome in ASSERTIVE:
                silent_errors.append(case.case_id)

        # citation validity: exists, admissible, and actually bears on the claim
        for cid in answer.cited:
            cited_total += 1
            node = case.graph.get(cid)
            if node is None or node.kind != "observation":
                continue
            if evidence_defects(case.graph, node, policy):  # type: ignore[arg-type]
                continue
            if polarity(case.graph.claim, node) == "irrelevant":  # type: ignore[arg-type]
                continue
            cited_valid += 1

        if gold.evidence:
            hit = len(set(answer.cited) & set(gold.evidence))
            recall_scores.append(hit / len(gold.evidence))

        if gold.outcome in ABSTAINING:
            abstain_cases += 1
            if answer.outcome in ABSTAINING:
                abstain_ok += 1
        else:
            decide_cases += 1
            if answer.outcome in ABSTAINING:
                over_abstained += 1

        if gold.conflicts:
            conflict_cases += 1
            if set(gold.conflicts) <= set(answer.reported_conflicts):
                conflict_found += 1

        if answer.unsupported_claim:
            unsupported += 1

    return Metrics(
        system=system,
        n=n,
        outcome_accuracy=correct / n if n else 0.0,
        silent_error_rate=len(silent_errors) / n if n else 0.0,
        citation_validity=cited_valid / cited_total if cited_total else 1.0,
        citation_recall=sum(recall_scores) / len(recall_scores)
        if recall_scores
        else 1.0,
        correct_abstention=abstain_ok / abstain_cases if abstain_cases else 1.0,
        over_abstention=over_abstained / decide_cases if decide_cases else 0.0,
        conflict_recall=conflict_found / conflict_cases if conflict_cases else 1.0,
        unsupported_claim_rate=unsupported / n if n else 0.0,
        silent_errors=silent_errors,
        wrong=wrong,
    )


def corpus_index(cases: Iterable[Case]) -> dict[str, str]:
    """observation id -> case id, so a cross-case citation names its real home."""
    index: dict[str, str] = {}
    for case in cases:
        for node_id in case.graph.nodes:
            index[node_id] = case.case_id
    return index
