"""The comparison system: same case, same reasoner, flattened context.

The baseline is given every field the agent can reach - nothing is withheld.
What it does not get is structure: no typed provenance to traverse, no
admissibility annotation, no case scoping beyond "these records arrived
together", and no validator between its answer and the record. The difference
between the two columns of the evaluation is therefore attributable to the
context model and the gate, not to a difference in information or in model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pv.graph import flatten_for_baseline
from pv.llm import AnthropicModel, Model, StubModel, naive_judgement
from pv.loader import Case
from pv.policy import polarity
from pv.types import Artifact, Observation

BASELINE_SYSTEM = """\
You review site-inspection records and decide whether a claim is supported.
Answer with a JSON object only:
{"outcome": "supported" | "contradicted" | "insufficient_evidence" |
 "needs_human_review", "cited": ["<record ids>"], "summary": "<one sentence>"}"""

BASELINE_PROMPT = """\
Claim: {claim_text}

Case records:
{blob}

Return the JSON object described in your instructions."""


@dataclass
class BaselineResult:
    case_id: str
    outcome: str
    cited: tuple[str, ...] = ()
    summary: str = ""
    model_name: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def run_baseline(case: Case, model: Model) -> BaselineResult:
    if isinstance(model, StubModel):
        return _run_stub_baseline(case, model)
    if isinstance(model, AnthropicModel):
        return _run_llm_baseline(case, model)
    raise TypeError(f"no baseline path for {type(model).__name__}")


def _run_stub_baseline(case: Case, model: StubModel) -> BaselineResult:
    """The same naive reasoner the agent stub uses, over undifferentiated records."""
    claim = case.graph.claim
    candidates: list[dict[str, Any]] = []
    for obs in case.graph.observations():
        candidates.append(
            {
                "id": obs.id,
                "label": obs.label,
                "value": obs.value,
                "confidence": obs.confidence,
                # Polarity is a property of the label, not of provenance, so
                # the baseline gets it too. It is admissibility, scoping, and
                # lineage that are absent here.
                "bears_on_claim": polarity(claim, obs),
            }
        )
    outcome, cited = naive_judgement(candidates)
    return BaselineResult(
        case_id=case.case_id,
        outcome=outcome,
        cited=tuple(cited),
        summary=f"Records indicate the claim is {outcome.replace('_', ' ')}.",
        model_name=f"baseline:{model.name}",
        raw={"candidates": len(candidates)},
    )


def _run_llm_baseline(case: Case, model: AnthropicModel) -> BaselineResult:
    blob = json.dumps(flatten_for_baseline(case.graph), indent=2, default=str)
    payload = model.complete_json(
        BASELINE_SYSTEM,
        BASELINE_PROMPT.format(claim_text=case.graph.claim.text, blob=blob),
    )
    return BaselineResult(
        case_id=case.case_id,
        outcome=payload.get("outcome", "insufficient_evidence"),
        cited=tuple(payload.get("cited", ())),
        summary=payload.get("summary", ""),
        model_name=f"baseline:{model.name}",
        raw=payload,
    )
