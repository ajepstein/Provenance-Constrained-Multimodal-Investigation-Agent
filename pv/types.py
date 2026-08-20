"""Core node, edge, and decision types.

This module is the schema. It imports nothing else from `pv` so every other
module can depend on it. All types are frozen: once a node is in the graph it
is never mutated. Corrections are represented as *new* nodes plus a
`supersedes` edge, so a contradiction is always still visible in the record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# --------------------------------------------------------------------------
# Outcomes
# --------------------------------------------------------------------------

Outcome = Literal[
    "supported",
    "contradicted",
    "insufficient_evidence",
    "needs_human_review",
]

OUTCOMES: tuple[Outcome, ...] = (
    "supported",
    "contradicted",
    "insufficient_evidence",
    "needs_human_review",
)

# Conservatism lattice. The validator may only move a decision *up* this
# ranking, never down: it can weaken a conclusion but never strengthen one.
CONSERVATISM: dict[str, int] = {
    "supported": 0,
    "contradicted": 0,
    "insufficient_evidence": 1,
    "needs_human_review": 2,
}


def most_conservative(a: Outcome, b: Outcome) -> Outcome:
    """Return whichever of two outcomes concedes more."""
    return a if CONSERVATISM[a] >= CONSERVATISM[b] else b


# --------------------------------------------------------------------------
# Graph nodes
# --------------------------------------------------------------------------

NodeKind = Literal[
    "artifact", "transformation", "observation", "claim", "tool_call", "decision"
]


@dataclass(frozen=True)
class Producer:
    """Who or what produced an observation."""

    type: Literal["model", "human", "nlp"]
    id: str
    version: str | None = None

    def label(self) -> str:
        return f"{self.id}@{self.version}" if self.version else self.id


@dataclass(frozen=True)
class Artifact:
    """A source file or record: an image, an operator note, an inspection form."""

    id: str
    case_id: str
    media_type: Literal["image", "operator_note", "inspection_record"]
    uri: str
    sha256: str | None = None
    captured_at: str | None = None  # ISO 8601 UTC; None => unknown, never guessed
    captured_by: str | None = None  # device / camera id
    uploaded_by: str | None = None
    site: str | None = None
    text: str | None = None  # verbatim content for note/record artifacts
    kind: NodeKind = "artifact"


@dataclass(frozen=True)
class Transformation:
    """An operation that produced one artifact from others."""

    id: str
    case_id: str
    op: str  # crop | resize | reencode | redact_exif | rotate | ocr
    tool: str
    tool_version: str
    at: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    # True when the op can remove or alter the evidence a detector would rely
    # on (a crop that drops the torso region, a redaction, an EXIF strip).
    evidence_affecting: bool = False
    kind: NodeKind = "transformation"


@dataclass(frozen=True)
class Observation:
    """A claim-bearing statement about an artifact.

    `label`/`value` are the machine-comparable core (e.g. hard_hat/present);
    `text` and `span` carry the human-readable provenance for assertions
    lifted out of free-text notes.
    """

    id: str
    case_id: str
    obs_kind: Literal["detection", "assertion", "metadata"]
    label: str
    value: str  # present | absent | unknown | free value for metadata
    producer: Producer
    confidence: float | None = None
    at: str | None = None
    text: str | None = None
    span: tuple[int, int] | None = None  # char offsets into the source artifact
    kind: NodeKind = "observation"


@dataclass(frozen=True)
class Claim:
    """The proposition under evaluation, plus what would count as evidence."""

    id: str
    case_id: str
    text: str
    predicate: str
    # Labels whose value decides the predicate. Anything else is off-topic.
    deciding_labels: tuple[str, ...]
    # All labels that must be observed `present` for the claim to hold.
    required_labels: tuple[str, ...]
    requires: tuple[str, ...]  # evidence-type requirements, see policy.py
    window_start: str
    window_end: str
    site: str
    kind: NodeKind = "claim"


Node = Artifact | Transformation | Observation | Claim


# --------------------------------------------------------------------------
# Edges
# --------------------------------------------------------------------------

# obs  --derived_from--> artifact       (an observation is *about* an artifact)
# art  --input_to------> transformation (an artifact fed an operation)
# tf   --produced------> artifact       (an operation emitted an artifact)
# art  --supersedes----> artifact       (a corrected re-upload replaces an older file)
# dec  --cites---------> observation    (a decision leans on an observation)
Relation = Literal["derived_from", "input_to", "produced", "supersedes", "cites"]


@dataclass(frozen=True)
class Edge:
    src: str
    rel: Relation
    dst: str
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Agent output
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Assertion:
    """One atomic statement in the agent's rationale, with its backing.

    Free prose is not accepted as a rationale. Every claim the agent makes
    must be an (predicate, value) pair carrying the observation ids that back
    it, which makes "unsupported claim" a decidable property rather than a
    judgement call.
    """

    statement: str
    cited: tuple[str, ...]


@dataclass(frozen=True)
class UnusedEvidence:
    id: str
    reason: str


@dataclass(frozen=True)
class Conflict:
    """Two in-scope observations that cannot both be true.

    Conflicts are computed from the graph, never taken from the agent, so a
    model cannot make one disappear by declining to mention it.
    """

    label: str
    positive: str  # observation id asserting `present`
    negative: str  # observation id asserting `absent`
    note: str = ""


@dataclass(frozen=True)
class Decision:
    case_id: str
    claim_id: str
    outcome: Outcome
    assertions: tuple[Assertion, ...] = ()
    evidence_used: tuple[str, ...] = ()
    evidence_not_used: tuple[UnusedEvidence, ...] = ()
    summary: str = ""


Severity = Literal["reject", "downgrade", "escalate"]


@dataclass(frozen=True)
class Violation:
    code: str
    severity: Severity
    message: str
    refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolCall:
    seq: int
    name: str
    arguments: dict[str, Any]
    result_digest: str
    result_ids: tuple[str, ...] = ()
