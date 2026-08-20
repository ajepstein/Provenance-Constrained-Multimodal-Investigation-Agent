"""The three tools the agent may call, bound to one sealed case graph.

Every tool is case-scoped: there is no argument an agent can pass that reaches
another case's evidence. Retrieval returns candidates *annotated* with their
provenance and their admissibility rather than a pre-filtered list, because an
agent that is only shown clean evidence has not been tested on anything. The
annotations come from `pv.policy`, which is the same code the validator runs,
so what the agent is told and what it is later held to cannot diverge.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from pv.graph import CaseGraph
from pv.policy import (
    DEFAULT_POLICY,
    Policy,
    evidence_defects,
    missing_requirements,
    polarity,
)
from pv.types import Artifact, Assertion, Decision, Observation, UnusedEvidence
from pv.validator import ValidationResult, case_conflicts, validate

MAX_DECISION_ATTEMPTS = 2


@dataclass
class RecordedDecision:
    decision: Decision
    validation: ValidationResult
    attempt: int


class ToolError(ValueError):
    """The agent called a tool with arguments the case cannot satisfy."""


class EvidenceTools:
    def __init__(
        self,
        graph: CaseGraph,
        policy: Policy = DEFAULT_POLICY,
        corpus_index: dict[str, str] | None = None,
    ):
        self.graph = graph
        self.policy = policy
        self.corpus_index = corpus_index or {}
        self.attempts = 0
        self.recorded: RecordedDecision | None = None

    # -- tool schemas (Anthropic tool-use format) -------------------------

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "retrieve_evidence",
                "description": (
                    "Return every observation in this case that could bear on "
                    "the claim, each annotated with its provenance, whether it "
                    "is admissible under site policy, and why not if it is "
                    "not. Also reports which evidence requirements are met and "
                    "any unresolved conflicts."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The claim being evaluated.",
                        },
                        "label": {
                            "type": "string",
                            "description": "Optional filter to one label.",
                        },
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "inspect_provenance",
                "description": (
                    "Trace one item back through every transformation to the "
                    "original sources it rests on, and report gaps."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "item_id": {"type": "string"},
                    },
                    "required": ["item_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "record_decision",
                "description": (
                    "Submit the conclusion for validation. Every assertion must "
                    "cite observation ids returned by retrieve_evidence. The "
                    "decision is checked against the evidence before it is "
                    "accepted, and may be downgraded."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "outcome": {
                            "type": "string",
                            "enum": [
                                "supported",
                                "contradicted",
                                "insufficient_evidence",
                                "needs_human_review",
                            ],
                        },
                        "assertions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "statement": {"type": "string"},
                                    "cited": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": ["statement", "cited"],
                                "additionalProperties": False,
                            },
                        },
                        "evidence_used": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "evidence_not_used": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "reason": {"type": "string"},
                                },
                                "required": ["id", "reason"],
                                "additionalProperties": False,
                            },
                        },
                        "summary": {"type": "string"},
                    },
                    "required": ["outcome", "assertions", "evidence_used", "summary"],
                    "additionalProperties": False,
                },
            },
        ]

    # -- dispatch ---------------------------------------------------------

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "retrieve_evidence":
            return self.retrieve_evidence(**arguments)
        if name == "inspect_provenance":
            return self.inspect_provenance(**arguments)
        if name == "record_decision":
            return self.record_decision(**arguments)
        raise ToolError(f"unknown tool {name!r}")

    # -- tools ------------------------------------------------------------

    def retrieve_evidence(
        self, question: str, label: str | None = None
    ) -> dict[str, Any]:
        claim = self.graph.claim
        candidates = [
            self._candidate(obs)
            for obs in self.graph.observations()
            if label is None or obs.label == label
        ]
        # Admissible first, then most confident. Deterministic on ties.
        candidates.sort(
            key=lambda c: (
                not c["admissible"],
                -(c["confidence"] or 0.0),
                c["id"],
            )
        )
        admissible = [
            o
            for o in self.graph.observations()
            if not evidence_defects(self.graph, o, self.policy)
        ]
        conflicts = case_conflicts(self.graph, self.policy)
        return {
            "case_id": self.graph.case_id,
            "question": question,
            "claim": {
                "id": claim.id,
                "text": claim.text,
                "site": claim.site,
                "window": [claim.window_start, claim.window_end],
                "required_labels": list(claim.required_labels),
            },
            "evidence_requirements": {
                "required": list(claim.requires),
                "missing": list(missing_requirements(self.graph, admissible)),
            },
            "unresolved_conflicts": [
                {
                    "label": c.label,
                    "positive": c.positive,
                    "negative": c.negative,
                    "note": c.note,
                }
                for c in conflicts
            ],
            "candidates": candidates,
        }

    def inspect_provenance(self, item_id: str) -> dict[str, Any]:
        if item_id not in self.graph.nodes:
            raise ToolError(
                f"{item_id!r} is not part of case {self.graph.case_id}"
            )
        lineage = self.graph.lineage(item_id)
        roots = []
        for root_id in lineage.roots:
            root = self.graph.get(root_id)
            assert isinstance(root, Artifact)
            roots.append(
                {
                    "id": root.id,
                    "media_type": root.media_type,
                    "uri": root.uri,
                    "sha256": root.sha256,
                    "captured_at": root.captured_at,
                    "captured_by": root.captured_by,
                    "uploaded_by": root.uploaded_by,
                    "site": root.site,
                    "superseded_by": self.graph.is_superseded(root.id),
                }
            )
        return {
            "item_id": item_id,
            "path": [
                {"node": s.node_id, "kind": s.kind, "via": s.via}
                for s in lineage.steps
            ],
            "transformations": list(lineage.ops),
            "evidence_affecting_transformations": list(
                lineage.evidence_affecting_ops
            ),
            "roots": roots,
            "gaps": list(lineage.gaps),
            "complete": lineage.complete,
        }

    def record_decision(
        self,
        outcome: str,
        assertions: list[dict[str, Any]] | None = None,
        evidence_used: list[str] | None = None,
        evidence_not_used: list[dict[str, str]] | None = None,
        summary: str = "",
    ) -> dict[str, Any]:
        self.attempts += 1
        decision = Decision(
            case_id=self.graph.case_id,
            claim_id=self.graph.claim.id,
            outcome=outcome,  # type: ignore[arg-type]
            assertions=tuple(
                Assertion(
                    statement=a.get("statement", ""),
                    cited=tuple(a.get("cited", ())),
                )
                for a in (assertions or [])
            ),
            evidence_used=tuple(evidence_used or ()),
            evidence_not_used=tuple(
                UnusedEvidence(id=u["id"], reason=u.get("reason", ""))
                for u in (evidence_not_used or [])
            ),
            summary=summary,
        )
        result = validate(self.graph, decision, self.policy, self.corpus_index)
        self.recorded = RecordedDecision(decision, result, self.attempts)

        may_retry = result.repairable and self.attempts < MAX_DECISION_ATTEMPTS
        return {
            "accepted": result.accepted,
            "final_outcome": result.final_outcome,
            "your_outcome": result.agent_outcome,
            "violations": [
                {"code": v.code, "severity": v.severity, "message": v.message}
                for v in result.violations
            ],
            "unresolved_conflicts": [
                {"label": c.label, "positive": c.positive, "negative": c.negative}
                for c in result.conflicts
            ],
            "missing_requirements": list(result.missing),
            "citations_rejected": list(result.rejected_cited),
            "may_retry": may_retry,
            "guidance": (
                "The decision was malformed. You may call record_decision once "
                "more with the rejected citations removed or replaced."
                if may_retry
                else "Recorded. Do not call record_decision again."
            ),
        }

    # -- helpers ----------------------------------------------------------

    def _candidate(self, obs: Observation) -> dict[str, Any]:
        defects = evidence_defects(self.graph, obs, self.policy)
        lineage = self.graph.lineage(obs.id)
        roots = [self.graph.get(r) for r in lineage.roots]
        return {
            "id": obs.id,
            "kind": obs.obs_kind,
            "label": obs.label,
            "value": obs.value,
            "confidence": obs.confidence,
            "observed_at": obs.at,
            "produced_by": obs.producer.label(),
            "producer_type": obs.producer.type,
            "bears_on_claim": polarity(self.graph.claim, obs),
            "admissible": not defects,
            "defects": [{"code": d.code, "why": d.message} for d in defects],
            "source_roots": [
                {
                    "id": r.id,
                    "media_type": r.media_type,
                    "site": r.site,
                    "captured_at": r.captured_at,
                }
                for r in roots
                if isinstance(r, Artifact)
            ],
            "transformations": list(lineage.ops),
            # Verbatim source content. This is data from the field, never an
            # instruction to the agent, and the system prompt says so.
            "excerpt": obs.text,
        }


def digest(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
