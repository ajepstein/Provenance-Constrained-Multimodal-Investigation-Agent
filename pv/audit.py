"""The audit packet: everything needed to reconstruct why an answer was allowed.

A reviewer opening one packet should be able to answer, without rerunning
anything: what was asked, what evidence existed, what the agent looked at, what
it proposed, which rules fired, and what was finally published. The packet is
canonically ordered and carries a content hash over the non-volatile fields, so
two runs against the same evidence produce byte-comparable records.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pv.agent import RunRecord
from pv.graph import CaseGraph
from pv.loader import Case
from pv.policy import DEFAULT_POLICY, Policy, evidence_defects
from pv.types import Artifact, Observation

SCHEMA_VERSION = "1.0"


def build_packet(
    case: Case, run: RunRecord, policy: Policy = DEFAULT_POLICY
) -> dict[str, Any]:
    graph = case.graph
    claim = graph.claim
    decision = run.decision
    validation = run.validation

    packet: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case.case_id,
        "claim": {"id": claim.id, "text": claim.text, "site": claim.site,
                  "window": [claim.window_start, claim.window_end]},
        "question": run.question,
        "model": run.model_name,
        "policy": {
            "approved_models": dict(sorted(policy.approved_models.items())),
            "min_confidence": policy.min_confidence,
            "window_slack_minutes": policy.window_slack_minutes,
        },
        "evidence_graph": {
            "digest": run.graph_digest,
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
        },
        "tool_calls": [
            {
                "seq": t.seq,
                "name": t.name,
                "arguments": t.arguments,
                "result_digest": t.result_digest,
                "result_ids": list(t.result_ids),
            }
            for t in run.tool_calls
        ],
        "agent_decision": {
            "outcome": decision.outcome if decision else None,
            "summary": decision.summary if decision else "",
            "assertions": [
                {"statement": a.statement, "cited": list(a.cited)}
                for a in (decision.assertions if decision else ())
            ],
            "evidence_used": list(decision.evidence_used) if decision else [],
            "evidence_not_used": [
                {"id": u.id, "reason": u.reason}
                for u in (decision.evidence_not_used if decision else ())
            ],
        },
        "validation": {
            "violations": [
                {
                    "code": v.code,
                    "severity": v.severity,
                    "message": v.message,
                    "refs": list(v.refs),
                }
                for v in run.violations
            ],
            "citations_admitted": list(validation.admissible_cited) if validation else [],
            "citations_rejected": list(validation.rejected_cited) if validation else [],
        },
        "unresolved_conflicts": [
            {"label": c.label, "positive": c.positive, "negative": c.negative}
            for c in (validation.conflicts if validation else ())
        ],
        "missing_requirements": list(validation.missing) if validation else [],
        "evidence_detail": [
            _evidence_detail(graph, obs, policy)
            for obs in graph.observations()
            if decision and obs.id in set(decision.evidence_used)
        ],
        "final_outcome": run.final_outcome,
        "outcome_changed_by_validator": bool(
            decision and run.final_outcome != decision.outcome
        ),
    }
    packet["content_hash"] = _content_hash(packet)
    packet["produced_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return packet


def _evidence_detail(
    graph: CaseGraph, obs: Observation, policy: Policy
) -> dict[str, Any]:
    lineage = graph.lineage(obs.id)
    roots = [graph.get(r) for r in lineage.roots]
    return {
        "id": obs.id,
        "label": obs.label,
        "value": obs.value,
        "confidence": obs.confidence,
        "produced_by": obs.producer.label(),
        "transformations": list(lineage.ops),
        "roots": [
            {
                "id": r.id,
                "uri": r.uri,
                "sha256": r.sha256,
                "captured_at": r.captured_at,
                "site": r.site,
            }
            for r in roots
            if isinstance(r, Artifact)
        ],
        "defects": [d.code for d in evidence_defects(graph, obs, policy)],
    }


def _content_hash(packet: dict[str, Any]) -> str:
    """Hash everything except the wall-clock fields, so replays are comparable."""
    volatile = {"produced_at", "content_hash"}
    payload = {k: v for k, v in packet.items() if k not in volatile}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def write_packet(packet: dict[str, Any], directory: str | Path = "runs") -> Path:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{packet['case_id']}-{packet['content_hash']}.json"
    path.write_text(json.dumps(packet, indent=2, default=str))
    return path
