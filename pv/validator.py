"""The deterministic gate between an agent's proposal and an accepted answer.

Two properties define this module:

1. **Monotonicity.** The validator may only move an outcome *up* the
   conservatism lattice - supported/contradicted -> insufficient ->
   needs_human_review. It never strengthens a conclusion, never flips a
   direction, and never authors an answer of its own. If an agent gets the
   direction wrong, the validator turns a confident wrong answer into an
   abstention; it does not turn it into the right answer.

2. **Case-level escalation.** Conflicts and provenance breaks are computed
   from the graph over *all* admissible evidence, not from what the agent
   chose to cite. An agent cannot make a conflict disappear by not mentioning
   it, or dodge review by ignoring the broken half of the case.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pv.graph import CaseGraph
from pv.policy import (
    DEFAULT_POLICY,
    Policy,
    evidence_defects,
    missing_requirements,
    polarity,
)
from pv.types import (
    Conflict,
    Decision,
    Observation,
    Outcome,
    Violation,
    most_conservative,
)

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_WORD = re.compile(r"[a-z_][a-z0-9_\-]+")


@dataclass(frozen=True)
class ValidationResult:
    final_outcome: Outcome
    agent_outcome: Outcome
    violations: tuple[Violation, ...] = ()
    conflicts: tuple[Conflict, ...] = ()
    missing: tuple[str, ...] = ()
    admissible_cited: tuple[str, ...] = ()
    rejected_cited: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        """True when the agent's own outcome survived unchanged."""
        return self.final_outcome == self.agent_outcome and not self.violations

    @property
    def repairable(self) -> bool:
        """True when the decision is malformed in a way the agent could fix."""
        return any(v.severity == "reject" for v in self.violations)

    def codes(self) -> tuple[str, ...]:
        return tuple(sorted({v.code for v in self.violations}))


# --------------------------------------------------------------------------
# Case-level facts (independent of what the agent cited)
# --------------------------------------------------------------------------


def case_conflicts(
    graph: CaseGraph, policy: Policy = DEFAULT_POLICY
) -> tuple[Conflict, ...]:
    """Every unresolved present/absent disagreement among admissible evidence.

    Computed from the graph so it survives an agent that would rather not
    mention it. Inadmissible evidence cannot create a conflict - that is what
    makes case-013 (a superseded upload) a resolved disagreement rather than
    an escalation.
    """
    claim = graph.claim
    admissible = [
        o for o in graph.observations() if not evidence_defects(graph, o, policy)
    ]
    conflicts: list[Conflict] = []
    for label in claim.deciding_labels:
        positives = [o.id for o in admissible if o.label == label and o.value == "present"]
        negatives = [o.id for o in admissible if o.label == label and o.value == "absent"]
        for pos in positives:
            for neg in negatives:
                conflicts.append(
                    Conflict(
                        label=label,
                        positive=pos,
                        negative=neg,
                        note="both sources are admissible and in scope",
                    )
                )
    return tuple(conflicts)


def case_escalations(
    graph: CaseGraph, policy: Policy = DEFAULT_POLICY
) -> tuple[Violation, ...]:
    """Defects anywhere in the case that require a person regardless of the answer."""
    out: list[Violation] = []
    for obs in graph.observations():
        for defect in evidence_defects(graph, obs, policy):
            if defect.severity == "escalate":
                out.append(
                    Violation(
                        code=defect.code,
                        severity="escalate",
                        message=defect.message,
                        refs=(obs.id,),
                    )
                )
    return tuple(out)


def case_missing_requirements(
    graph: CaseGraph, policy: Policy = DEFAULT_POLICY
) -> tuple[str, ...]:
    admissible = [
        o for o in graph.observations() if not evidence_defects(graph, o, policy)
    ]
    return missing_requirements(graph, admissible)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def validate(
    graph: CaseGraph,
    decision: Decision,
    policy: Policy = DEFAULT_POLICY,
    corpus_index: dict[str, str] | None = None,
) -> ValidationResult:
    """Check a proposed decision and return the outcome that may be published.

    `corpus_index` maps observation id -> case id across the whole fixture set.
    It exists only to give a better message when an agent cites evidence from
    a different case: without it such an id is merely unresolvable.
    """
    claim = graph.claim
    violations: list[Violation] = []

    cited_ids: list[str] = list(decision.evidence_used)
    for assertion in decision.assertions:
        cited_ids.extend(assertion.cited)
    cited_ids = sorted(set(cited_ids))

    # -- R1: every citation resolves to an observation in this case --------
    resolved: list[Observation] = []
    for cid in cited_ids:
        node = graph.get(cid)
        if node is None:
            foreign = (corpus_index or {}).get(cid)
            if foreign:
                violations.append(
                    Violation(
                        "CITATION_FOREIGN_CASE",
                        "reject",
                        f"cited {cid!r} belongs to case {foreign}, not {graph.case_id}",
                        (cid,),
                    )
                )
            else:
                violations.append(
                    Violation(
                        "CITATION_UNRESOLVED",
                        "reject",
                        f"cited {cid!r} does not exist in the case graph",
                        (cid,),
                    )
                )
            continue
        if not isinstance(node, Observation):
            violations.append(
                Violation(
                    "CITATION_NOT_EVIDENCE",
                    "reject",
                    f"cited {cid!r} is a {node.kind}, not an observation",
                    (cid,),
                )
            )
            continue
        resolved.append(node)

    # -- R2: every citation is admissible under policy ---------------------
    admissible: list[Observation] = []
    rejected: list[str] = []
    for obs in resolved:
        defects = evidence_defects(graph, obs, policy)
        if not defects:
            admissible.append(obs)
            continue
        rejected.append(obs.id)
        for defect in defects:
            violations.append(
                Violation(
                    defect.code,
                    "escalate" if defect.severity == "escalate" else "downgrade",
                    defect.message,
                    (obs.id,),
                )
            )

    # -- R3: every assertion carries backing -------------------------------
    admissible_ids = {o.id for o in admissible}
    for assertion in decision.assertions:
        if not assertion.cited:
            violations.append(
                Violation(
                    "UNSUPPORTED_ASSERTION",
                    "reject",
                    f"assertion {assertion.statement!r} cites nothing",
                )
            )
        elif not set(assertion.cited) & admissible_ids:
            violations.append(
                Violation(
                    "UNSUPPORTED_ASSERTION",
                    "reject",
                    f"assertion {assertion.statement!r} rests entirely on "
                    f"evidence that was not admitted",
                    tuple(assertion.cited),
                )
            )

    # -- R4: the free-text summary introduces nothing new ------------------
    violations.extend(_check_summary_grounding(graph, decision, admissible))

    # -- case-level facts --------------------------------------------------
    conflicts = case_conflicts(graph, policy)
    escalations = case_escalations(graph, policy)
    missing = case_missing_requirements(graph, policy)
    violations.extend(escalations)

    if conflicts:
        for conflict in conflicts:
            violations.append(
                Violation(
                    "UNRESOLVED_CONFLICT",
                    "escalate",
                    f"{conflict.label}: {conflict.positive} says present while "
                    f"{conflict.negative} says absent; {conflict.note}",
                    (conflict.positive, conflict.negative),
                )
            )
    if missing:
        violations.append(
            Violation(
                "MISSING_REQUIRED_EVIDENCE",
                "downgrade",
                f"no admissible evidence satisfies: {', '.join(missing)}",
                tuple(missing),
            )
        )

    # -- R5: does the cited evidence actually point the way the agent said? -
    supporting = [o for o in admissible if polarity(claim, o) == "supports"]
    contradicting = [o for o in admissible if polarity(claim, o) == "contradicts"]

    if decision.outcome == "supported":
        uncovered = [
            label
            for label in claim.required_labels
            if not any(o.label == label for o in supporting)
        ]
        if uncovered:
            violations.append(
                Violation(
                    "REQUIRED_LABEL_UNCOVERED",
                    "downgrade",
                    f"no admissible cited evidence shows {', '.join(uncovered)} present",
                    tuple(uncovered),
                )
            )
        if contradicting:
            violations.append(
                Violation(
                    "OUTCOME_AGAINST_EVIDENCE",
                    "downgrade",
                    f"answer is supported while admissible cited evidence "
                    f"{', '.join(o.id for o in contradicting)} contradicts the claim",
                    tuple(o.id for o in contradicting),
                )
            )
    elif decision.outcome == "contradicted":
        on_required = [
            o for o in contradicting if o.label in claim.required_labels
        ]
        if not on_required:
            violations.append(
                Violation(
                    "OUTCOME_AGAINST_EVIDENCE",
                    "downgrade",
                    "answer is contradicted but no admissible cited evidence "
                    "shows a required item absent",
                )
            )

    # -- combine -----------------------------------------------------------
    final = _apply(decision.outcome, violations)

    return ValidationResult(
        final_outcome=final,
        agent_outcome=decision.outcome,
        violations=tuple(violations),
        conflicts=conflicts,
        missing=missing,
        admissible_cited=tuple(sorted(admissible_ids)),
        rejected_cited=tuple(sorted(rejected)),
    )


def _apply(agent_outcome: Outcome, violations: list[Violation]) -> Outcome:
    """Fold violations into an outcome, only ever conceding more."""
    outcome: Outcome = agent_outcome
    for violation in violations:
        if violation.severity == "escalate":
            outcome = most_conservative(outcome, "needs_human_review")
        elif violation.severity == "downgrade":
            outcome = most_conservative(outcome, "insufficient_evidence")
        elif violation.severity == "reject":
            # A malformed decision that the agent did not repair is not an
            # answer at all. It goes to a person rather than being guessed at.
            outcome = most_conservative(outcome, "needs_human_review")
    return outcome


def _check_summary_grounding(
    graph: CaseGraph, decision: Decision, admissible: list[Observation]
) -> list[Violation]:
    """Reject prose that asserts more than the citations carry.

    The check is deliberately narrow so it is decidable rather than a matter
    of taste: numbers must appear in the cited evidence or the claim, and a
    label from the case vocabulary may only be named if it was cited. That
    catches the two ways a rationale actually drifts - invented quantities and
    borrowed findings.
    """
    if not decision.summary:
        return []

    grounded_text = " ".join(
        [graph.claim.text]
        + [f"{o.label} {o.value} {o.confidence or ''} {o.text or ''}" for o in admissible]
    ).lower()

    violations: list[Violation] = []
    grounded_numbers = set(_NUMBER.findall(grounded_text))
    for number in _NUMBER.findall(decision.summary.lower()):
        if number not in grounded_numbers:
            violations.append(
                Violation(
                    "UNGROUNDED_NUMBER",
                    "reject",
                    f"the summary states {number!r}, which appears in no cited "
                    f"evidence",
                )
            )

    case_labels = {o.label for o in graph.observations()}
    cited_labels = {o.label for o in admissible}
    summary_words = set(_WORD.findall(decision.summary.lower()))
    for label in sorted(case_labels - cited_labels):
        if label.lower() in summary_words:
            violations.append(
                Violation(
                    "UNCITED_LABEL_MENTION",
                    "reject",
                    f"the summary discusses {label!r} without citing any "
                    f"admissible observation of it",
                    (label,),
                )
            )
    return violations
