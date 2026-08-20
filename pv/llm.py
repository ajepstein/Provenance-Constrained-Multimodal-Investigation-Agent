"""Models that can drive the agent loop.

Three implementations share one interface so the harness measures the harness,
not the model:

* `StubModel("naive")`   - follows the tool protocol but ignores the provenance
                           annotations it is given. The floor.
* `StubModel("careful")` - respects the annotations. The ceiling.
* `AdversarialModel`     - always answers "supported" and cites whatever is at
                           hand. Used to show what the gate blocks.
* `AnthropicModel`       - a real Claude tool-calling loop.

The stubs are deliberately naive reasoners, not stand-ins for Claude. Their
absolute accuracy says nothing about a real model's; what they establish is
which errors the deterministic layer catches *regardless* of the model, by
running the same reasoner on both sides of the comparison.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolUse:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    tool_use_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class Turn:
    text: str = ""
    tool_uses: tuple[ToolUse, ...] = ()


class Model(Protocol):
    name: str

    def start(self, system: str, question: str, tools: list[dict[str, Any]]) -> None: ...

    def next_turn(self) -> Turn: ...

    def submit_results(self, results: list[ToolResult]) -> None: ...


# --------------------------------------------------------------------------
# Deterministic stubs
# --------------------------------------------------------------------------


def naive_judgement(candidates: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Decide from a list of candidate records, the obvious way.

    Weigh whatever bears on the claim, break a tie by confidence, and cite the
    strongest few. This is the reasoning both the baseline and the agent get,
    so any difference between them comes from the context and the gate.
    """
    relevant = [c for c in candidates if c.get("bears_on_claim") != "irrelevant"]
    supports = [c for c in relevant if c.get("bears_on_claim") == "supports"]
    against = [c for c in relevant if c.get("bears_on_claim") == "contradicts"]

    def strength(items: list[dict[str, Any]]) -> float:
        return max((c.get("confidence") or 0.75) for c in items) if items else 0.0

    if not relevant:
        return "insufficient_evidence", []
    if supports and not against:
        chosen = supports
        outcome = "supported"
    elif against and not supports:
        chosen = against
        outcome = "contradicted"
    elif strength(supports) >= strength(against):
        chosen = supports
        outcome = "supported"
    else:
        chosen = against
        outcome = "contradicted"

    chosen = sorted(chosen, key=lambda c: (-(c.get("confidence") or 0.75), c["id"]))
    return outcome, [c["id"] for c in chosen]


@dataclass
class StubModel:
    """A scripted agent that follows the tool protocol without an API call."""

    strategy: str = "naive"
    name: str = field(init=False)
    _question: str = field(default="", init=False)
    _results: list[tuple[str, dict[str, Any]]] = field(default_factory=list, init=False)
    _pending: dict[str, str] = field(default_factory=dict, init=False)
    _counter: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.name = f"stub:{self.strategy}"

    def start(self, system: str, question: str, tools: list[dict[str, Any]]) -> None:
        self._question = question
        self._results = []
        self._pending = {}
        self._counter = 0

    def _use(self, name: str, arguments: dict[str, Any]) -> Turn:
        self._counter += 1
        use_id = f"call_{self._counter}"
        self._pending[use_id] = name
        return Turn(tool_uses=(ToolUse(use_id, name, arguments),))

    def _last(self, name: str) -> dict[str, Any] | None:
        for called, payload in reversed(self._results):
            if called == name:
                return payload
        return None

    def submit_results(self, results: list[ToolResult]) -> None:
        for result in results:
            name = self._pending.get(result.tool_use_id, "?")
            try:
                payload = json.loads(result.content)
            except json.JSONDecodeError:
                payload = {"error": result.content}
            self._results.append((name, payload))

    def next_turn(self) -> Turn:
        called = [name for name, _ in self._results]

        if "retrieve_evidence" not in called:
            return self._use("retrieve_evidence", {"question": self._question})

        retrieved = self._last("retrieve_evidence") or {}
        candidates: list[dict[str, Any]] = retrieved.get("candidates", [])

        if "inspect_provenance" not in called and candidates:
            return self._use("inspect_provenance", {"item_id": candidates[0]["id"]})

        recorded = self._last("record_decision")
        if recorded is None:
            return self._use("record_decision", self._decide(retrieved, candidates))
        if recorded.get("may_retry"):
            dropped = set(recorded.get("citations_rejected", ()))
            survivors = [c for c in candidates if c["id"] not in dropped]
            return self._use(
                "record_decision", self._decide(retrieved, survivors, repaired=True)
            )
        return Turn(text=f"Recorded: {recorded.get('final_outcome')}")

    def _decide(
        self,
        retrieved: dict[str, Any],
        candidates: list[dict[str, Any]],
        repaired: bool = False,
    ) -> dict[str, Any]:
        if self.strategy == "careful":
            return self._decide_carefully(retrieved, candidates, repaired)
        outcome, cited = naive_judgement(candidates)
        return self._payload(
            outcome, cited, candidates, self._summary(outcome), repaired
        )

    def _decide_carefully(
        self,
        retrieved: dict[str, Any],
        candidates: list[dict[str, Any]],
        repaired: bool = False,
    ) -> dict[str, Any]:
        """Use only admissible evidence, and read the claim as the conjunction
        it is.

        The claim asserts that *every* required item was present, so a single
        admissible absence contradicts it and a single uncovered item means the
        evidence does not reach a conclusion. The naive path instead weighs
        labels against each other by confidence, which is how a plausible
        reasoner ends up comparing a hard-hat detection with a vest detection
        and calling the louder one the answer.
        """
        pool = [c for c in candidates if c.get("admissible")]
        relevant = [c["id"] for c in pool if c.get("bears_on_claim") != "irrelevant"]
        required = retrieved.get("claim", {}).get("required_labels", [])

        if retrieved.get("unresolved_conflicts"):
            return self._payload(
                "needs_human_review",
                relevant,
                candidates,
                "Admissible sources disagree on a required item; escalating "
                "rather than choosing between them.",
                repaired,
            )
        if retrieved.get("evidence_requirements", {}).get("missing"):
            return self._payload(
                "insufficient_evidence",
                relevant,
                candidates,
                "Required evidence types are not satisfied by admissible sources.",
                repaired,
            )

        absent = [
            c for c in pool if c["label"] in required and c["value"] == "absent"
        ]
        if absent:
            return self._payload(
                "contradicted",
                [c["id"] for c in absent],
                candidates,
                self._summary("contradicted"),
                repaired,
            )

        present = {
            label: [
                c["id"] for c in pool
                if c["label"] == label and c["value"] == "present"
            ]
            for label in required
        }
        if all(present.values()):
            cited = sorted({i for ids in present.values() for i in ids})
            return self._payload(
                "supported", cited, pool, self._summary("supported"), repaired
            )
        return self._payload(
            "insufficient_evidence",
            relevant,
            pool,
            "Not every required item is shown present by admissible evidence.",
            repaired,
        )

    def _payload(
        self,
        outcome: str,
        cited: list[str],
        pool: list[dict[str, Any]],
        summary: str,
        repaired: bool = False,
    ) -> dict[str, Any]:
        by_id = {c["id"]: c for c in pool}
        assertions = [
            {
                "statement": f"{by_id[c]['label']} is {by_id[c]['value']}",
                "cited": [c],
            }
            for c in cited
            if c in by_id
        ]
        unused = [
            {
                "id": c["id"],
                "reason": (
                    "bears on a different question"
                    if c.get("bears_on_claim") == "irrelevant"
                    else "; ".join(d["code"] for d in c.get("defects", []))
                    or "not needed for this conclusion"
                ),
            }
            for c in pool
            if c["id"] not in set(cited)
        ]
        return {
            "outcome": outcome,
            "assertions": assertions,
            "evidence_used": cited,
            "evidence_not_used": unused,
            "summary": summary + (" (revised after validation)" if repaired else ""),
        }

    @staticmethod
    def _summary(outcome: str) -> str:
        return {
            "supported": "Cited evidence shows the required equipment present.",
            "contradicted": "Cited evidence shows a required item absent.",
            "insufficient_evidence": "The available evidence does not decide the claim.",
            "needs_human_review": "The case needs a person to resolve.",
        }[outcome]


@dataclass
class AdversarialModel(StubModel):
    """Answers "supported" no matter what, and cites loosely.

    Not a realistic model - a bound. Whatever this returns, the published
    outcome should still never be wrong in the unsafe direction.
    """

    strategy: str = "adversarial"
    foreign_id: str = "obs-004-a"

    def _decide(
        self,
        retrieved: dict[str, Any],
        candidates: list[dict[str, Any]],
        repaired: bool = False,
    ) -> dict[str, Any]:
        cited = [c["id"] for c in candidates][:3] + [self.foreign_id]
        return {
            "outcome": "supported",
            "assertions": [
                {"statement": "all required equipment was present", "cited": cited}
            ],
            "evidence_used": cited,
            "evidence_not_used": [],
            "summary": "All 7 workers were in full PPE for the entire shift.",
        }


# --------------------------------------------------------------------------
# Claude
# --------------------------------------------------------------------------

DEFAULT_MODEL_ID = "claude-opus-5"


class AnthropicModel:
    """A real Claude tool-calling loop.

    The agent contract is identical to the stubs': the model may only reach
    evidence through the tools, and `record_decision` is still the gate.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, max_tokens: int = 8000):
        import anthropic  # imported lazily: the offline path needs no SDK

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Run with --llm stub for the "
                "offline, deterministic path."
            )
        self._anthropic = anthropic
        self._client = anthropic.Anthropic()
        self.model_id = model_id
        self.name = f"anthropic:{model_id}"
        self.max_tokens = max_tokens
        self._system = ""
        self._tools: list[dict[str, Any]] = []
        self._messages: list[dict[str, Any]] = []
        self._last: Any = None

    def start(self, system: str, question: str, tools: list[dict[str, Any]]) -> None:
        self._system = system
        self._tools = tools
        self._messages = [{"role": "user", "content": question}]

    def next_turn(self) -> Turn:
        response = self._client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            system=self._system,
            tools=self._tools,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            messages=self._messages,
        )
        self._last = response
        # Echo the assistant turn back verbatim, thinking blocks included.
        self._messages.append({"role": "assistant", "content": response.content})
        text = "".join(b.text for b in response.content if b.type == "text")
        uses = tuple(
            ToolUse(id=b.id, name=b.name, input=dict(b.input))
            for b in response.content
            if b.type == "tool_use"
        )
        return Turn(text=text, tool_uses=uses)

    def submit_results(self, results: list[ToolResult]) -> None:
        self._messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": r.tool_use_id,
                        "content": r.content,
                        **({"is_error": True} if r.is_error else {}),
                    }
                    for r in results
                ],
            }
        )

    def complete_json(self, system: str, prompt: str) -> dict[str, Any]:
        """One-shot call used by the flattened-context baseline."""
        response = self._client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            system=system,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        return parse_json_object(text)


def parse_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def build_model(spec: str) -> Model:
    """Resolve a --llm flag to a model instance."""
    if spec in ("stub", "naive"):
        return StubModel("naive")
    if spec == "careful":
        return StubModel("careful")
    if spec == "adversarial":
        return AdversarialModel()
    if spec.startswith("anthropic"):
        _, _, model_id = spec.partition(":")
        return AnthropicModel(model_id or DEFAULT_MODEL_ID)
    raise ValueError(
        f"unknown model {spec!r}; expected stub | careful | adversarial | "
        f"anthropic[:model-id]"
    )
