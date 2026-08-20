"""Evidence policy: what counts as admissible evidence, and why it might not.

This module holds the single implementation of evidence admissibility. Both
the agent-facing tools and the validator call it, so what the agent is shown
about a piece of evidence and what the validator later enforces cannot drift
apart. If they could, the agent would be able to "win" by citing something
that looked fine at retrieval time and was rejected at decision time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from pv.graph import CaseGraph
from pv.types import Artifact, Claim, Observation

# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Policy:
    """Site rules. Deliberately small and specific to this fixture."""

    # detector name -> minimum approved version
    approved_models: dict[str, str]
    min_confidence: float = 0.60
    # how far outside the claim window a capture may fall and still count
    window_slack_minutes: int = 0
    # a detection that ran downstream of a lossy, evidence-affecting op is not
    # admissible on its own — the operation may have removed what it looked for
    reject_evidence_affecting_lineage: bool = True


DEFAULT_POLICY = Policy(
    approved_models={"ppe-detector": "2.3.0"},
    min_confidence=0.60,
)


# Evidence-type requirements a claim can ask for.
REQUIREMENT_TYPES = ("image_evidence", "detector_evidence", "human_record")


@dataclass(frozen=True)
class Defect:
    code: str
    message: str
    # block: the observation cannot count as evidence.
    # escalate: it cannot count *and* a human should look at the case.
    severity: str = "block"


def _parse(ts: str | None) -> datetime | None:
    if ts is None:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _version(v: str) -> tuple[int, ...]:
    return tuple(int(part) for part in v.split(".") if part.isdigit())


# --------------------------------------------------------------------------
# Admissibility
# --------------------------------------------------------------------------


def evidence_defects(
    graph: CaseGraph, obs: Observation, policy: Policy = DEFAULT_POLICY
) -> tuple[Defect, ...]:
    """Everything wrong with using `obs` as evidence for this case's claim.

    An empty tuple means admissible. Defects are *accumulated*, not
    short-circuited: a reviewer should see every reason at once.
    """
    claim: Claim = graph.claim
    defects: list[Defect] = []
    lineage = graph.lineage(obs.id)

    # -- provenance completeness ------------------------------------------
    hard_gaps = [g for g in lineage.gaps if "capture time unknown" not in g]
    if hard_gaps:
        defects.append(
            Defect(
                "BROKEN_PROVENANCE",
                f"{obs.id}: source lineage is incomplete ({'; '.join(hard_gaps)})",
                severity="escalate",
            )
        )
    if not lineage.roots:
        defects.append(
            Defect(
                "BROKEN_PROVENANCE",
                f"{obs.id}: does not trace back to any original source",
                severity="escalate",
            )
        )

    roots: list[Artifact] = [
        n for n in (graph.get(r) for r in lineage.roots) if isinstance(n, Artifact)
    ]

    # -- scope: does this evidence even belong to the claimed subject? -----
    off_site = [r.id for r in roots if r.site is not None and r.site != claim.site]
    if off_site:
        defects.append(
            Defect(
                "SCOPE_MISMATCH",
                f"{obs.id}: derives from {', '.join(off_site)} at a different "
                f"site than the claim ({claim.site})",
            )
        )

    # -- time: inside the inspection window? ------------------------------
    start = _parse(claim.window_start)
    end = _parse(claim.window_end)
    slack = timedelta(minutes=policy.window_slack_minutes)
    for root in roots:
        captured = _parse(root.captured_at)
        if captured is None:
            defects.append(
                Defect(
                    "CAPTURE_TIME_UNKNOWN",
                    f"{obs.id}: {root.id} has no capture time, so it cannot be "
                    f"placed inside the inspection window",
                )
            )
        elif start and end and not (start - slack <= captured <= end + slack):
            defects.append(
                Defect(
                    "OUT_OF_WINDOW",
                    f"{obs.id}: {root.id} was captured {root.captured_at}, "
                    f"outside the window {claim.window_start}..{claim.window_end}",
                )
            )

    # -- supersession: was the source corrected after the fact? -----------
    for step in lineage.steps:
        replacement = graph.is_superseded(step.node_id)
        if replacement is not None:
            defects.append(
                Defect(
                    "SUPERSEDED_SOURCE",
                    f"{obs.id}: rests on {step.node_id}, which was superseded "
                    f"by {replacement}",
                )
            )

    # -- transformations that may have removed the evidence ---------------
    if policy.reject_evidence_affecting_lineage and lineage.evidence_affecting_ops:
        defects.append(
            Defect(
                "EVIDENCE_AFFECTING_TRANSFORM",
                f"{obs.id}: produced downstream of "
                f"{', '.join(lineage.evidence_affecting_ops)}, which may have "
                f"removed the region the observation depends on",
            )
        )

    # -- model version and confidence -------------------------------------
    if obs.obs_kind == "detection":
        producer = obs.producer
        minimum = policy.approved_models.get(producer.id)
        if minimum is None:
            defects.append(
                Defect(
                    "UNAPPROVED_MODEL",
                    f"{obs.id}: produced by {producer.id}, which is not an "
                    f"approved detector",
                )
            )
        elif producer.version is None or _version(producer.version) < _version(minimum):
            defects.append(
                Defect(
                    "UNAPPROVED_MODEL_VERSION",
                    f"{obs.id}: {producer.label()} is below the approved "
                    f"minimum {producer.id}@{minimum}",
                )
            )
        if obs.confidence is not None and obs.confidence < policy.min_confidence:
            defects.append(
                Defect(
                    "LOW_CONFIDENCE",
                    f"{obs.id}: confidence {obs.confidence:.2f} is below the "
                    f"floor of {policy.min_confidence:.2f}",
                )
            )

    return tuple(defects)


def is_admissible(
    graph: CaseGraph, obs: Observation, policy: Policy = DEFAULT_POLICY
) -> bool:
    return not evidence_defects(graph, obs, policy)


def admissible_observations(
    graph: CaseGraph, policy: Policy = DEFAULT_POLICY
) -> list[Observation]:
    return [o for o in graph.observations() if is_admissible(graph, o, policy)]


# --------------------------------------------------------------------------
# Relevance and polarity
# --------------------------------------------------------------------------


def polarity(claim: Claim, obs: Observation) -> str:
    """How this observation bears on the claim: supports, contradicts, or neither.

    Polarity is derived from the claim's declared deciding labels, not from a
    model's opinion, so it is reproducible and reviewable. Free-text notes get
    a polarity only once they have been lifted into a labelled assertion (see
    `pv/interpret.py`), and that lift is itself a provenance-tracked step.
    """
    if obs.label not in claim.deciding_labels:
        return "irrelevant"
    if obs.value == "present":
        return "supports"
    if obs.value == "absent":
        return "contradicts"
    return "irrelevant"


def satisfied_requirements(
    graph: CaseGraph, observations: list[Observation]
) -> set[str]:
    """Which of the claim's evidence-type requirements these observations meet."""
    met: set[str] = set()
    for obs in observations:
        roots = [
            n for n in (graph.get(r) for r in graph.lineage(obs.id).roots)
            if isinstance(n, Artifact)
        ]
        if any(r.media_type == "image" for r in roots):
            met.add("image_evidence")
        if any(
            r.media_type in ("operator_note", "inspection_record") for r in roots
        ):
            met.add("human_record")
        if obs.obs_kind == "detection":
            met.add("detector_evidence")
    return met


def missing_requirements(
    graph: CaseGraph, observations: list[Observation]
) -> tuple[str, ...]:
    met = satisfied_requirements(graph, observations)
    return tuple(r for r in graph.claim.requires if r not in met)
