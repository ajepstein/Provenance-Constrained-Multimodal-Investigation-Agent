"""Load case fixtures from YAML into sealed graphs.

The loader is strict on purpose: an unknown key or a dangling reference is an
error at load time, not a surprise at decision time. Fixtures are the contract
between the dataset and every downstream component, so they get validated once,
loudly, in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pv.graph import CaseGraph, GraphError
from pv.types import Artifact, Claim, Observation, Producer, Transformation

_TOP_KEYS = {
    "case_id", "title", "tags", "claim", "artifacts", "transformations",
    "observations", "supersedes", "expected",
}
_CLAIM_KEYS = {
    "id", "text", "predicate", "deciding_labels", "required_labels", "requires",
    "window_start", "window_end", "site",
}
_ARTIFACT_KEYS = {
    "id", "media_type", "uri", "sha256", "captured_at", "captured_by",
    "uploaded_by", "site", "text",
}
_TRANSFORM_KEYS = {
    "id", "op", "tool", "tool_version", "at", "params", "evidence_affecting",
    "inputs", "outputs",
}
_OBSERVATION_KEYS = {
    "id", "obs_kind", "label", "value", "confidence", "at", "text", "span",
    "derived_from", "producer",
}
_EXPECTED_KEYS = {"outcome", "evidence", "conflicts", "missing", "rationale"}


class CaseLoadError(ValueError):
    """A fixture file is malformed. The message always names the file."""


@dataclass(frozen=True)
class Golden:
    """The hand-labelled expectation for one case."""

    case_id: str
    outcome: str
    evidence: tuple[str, ...]
    conflicts: tuple[str, ...]
    missing: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class Case:
    case_id: str
    title: str
    tags: tuple[str, ...]
    graph: CaseGraph
    golden: Golden
    path: Path


def _require(mapping: dict[str, Any], keys: set[str], where: str) -> None:
    unknown = set(mapping) - keys
    if unknown:
        raise CaseLoadError(f"{where}: unknown key(s) {sorted(unknown)}")


def _need(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise CaseLoadError(f"{where}: missing required key {key!r}")
    return mapping[key]


def load_case(path: str | Path) -> Case:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise CaseLoadError(f"{path}: invalid YAML ({exc})") from exc
    if not isinstance(raw, dict):
        raise CaseLoadError(f"{path}: file is not a YAML mapping")

    _require(raw, _TOP_KEYS, str(path))
    case_id = _need(raw, "case_id", str(path))
    graph = CaseGraph(case_id)

    # -- claim ------------------------------------------------------------
    claim_raw = _need(raw, "claim", str(path))
    _require(claim_raw, _CLAIM_KEYS, f"{path}: claim")
    graph.add_node(
        Claim(
            id=_need(claim_raw, "id", f"{path}: claim"),
            case_id=case_id,
            text=_need(claim_raw, "text", f"{path}: claim"),
            predicate=_need(claim_raw, "predicate", f"{path}: claim"),
            deciding_labels=tuple(claim_raw.get("deciding_labels", ())),
            required_labels=tuple(claim_raw.get("required_labels", ())),
            requires=tuple(_need(claim_raw, "requires", f"{path}: claim")),
            window_start=_need(claim_raw, "window_start", f"{path}: claim"),
            window_end=_need(claim_raw, "window_end", f"{path}: claim"),
            site=_need(claim_raw, "site", f"{path}: claim"),
        )
    )

    # -- artifacts --------------------------------------------------------
    for art in raw.get("artifacts") or []:
        _require(art, _ARTIFACT_KEYS, f"{path}: artifact {art.get('id')}")
        graph.add_node(
            Artifact(
                id=_need(art, "id", f"{path}: artifact"),
                case_id=case_id,
                media_type=_need(art, "media_type", f"{path}: artifact {art['id']}"),
                uri=_need(art, "uri", f"{path}: artifact {art['id']}"),
                sha256=art.get("sha256"),
                captured_at=art.get("captured_at"),
                captured_by=art.get("captured_by"),
                uploaded_by=art.get("uploaded_by"),
                site=art.get("site"),
                text=art.get("text"),
            )
        )

    # -- transformations --------------------------------------------------
    pending_edges: list[tuple[str, str, str]] = []
    for tf in raw.get("transformations") or []:
        _require(tf, _TRANSFORM_KEYS, f"{path}: transformation {tf.get('id')}")
        where = f"{path}: transformation {tf.get('id')}"
        graph.add_node(
            Transformation(
                id=_need(tf, "id", where),
                case_id=case_id,
                op=_need(tf, "op", where),
                tool=_need(tf, "tool", where),
                tool_version=_need(tf, "tool_version", where),
                at=tf.get("at"),
                params=tf.get("params") or {},
                evidence_affecting=bool(tf.get("evidence_affecting", False)),
            )
        )
        for src in tf.get("inputs") or []:
            pending_edges.append((src, "input_to", tf["id"]))
        for dst in tf.get("outputs") or []:
            pending_edges.append((tf["id"], "produced", dst))

    # -- observations -----------------------------------------------------
    for obs in raw.get("observations") or []:
        _require(obs, _OBSERVATION_KEYS, f"{path}: observation {obs.get('id')}")
        where = f"{path}: observation {obs.get('id')}"
        producer = _need(obs, "producer", where)
        span = obs.get("span")
        graph.add_node(
            Observation(
                id=_need(obs, "id", where),
                case_id=case_id,
                obs_kind=_need(obs, "obs_kind", where),
                label=_need(obs, "label", where),
                value=_need(obs, "value", where),
                producer=Producer(
                    type=_need(producer, "type", f"{where}: producer"),
                    id=_need(producer, "id", f"{where}: producer"),
                    version=producer.get("version"),
                ),
                confidence=obs.get("confidence"),
                at=obs.get("at"),
                text=obs.get("text"),
                span=tuple(span) if span else None,
            )
        )
        derived = _need(obs, "derived_from", where)
        for src in derived:
            pending_edges.append((obs["id"], "derived_from", src))

    for older, newer in (raw.get("supersedes") or {}).items():
        pending_edges.append((newer, "supersedes", older))

    for src, rel, dst in pending_edges:
        try:
            graph.add_edge(src, rel, dst)  # type: ignore[arg-type]
        except GraphError as exc:
            raise CaseLoadError(f"{path}: {exc}") from exc

    try:
        graph.seal()
    except GraphError as exc:
        raise CaseLoadError(f"{path}: {exc}") from exc

    # -- golden label -----------------------------------------------------
    exp = _need(raw, "expected", str(path))
    _require(exp, _EXPECTED_KEYS, f"{path}: expected")
    golden = Golden(
        case_id=case_id,
        outcome=_need(exp, "outcome", f"{path}: expected"),
        evidence=tuple(exp.get("evidence") or ()),
        conflicts=tuple(exp.get("conflicts") or ()),
        missing=tuple(exp.get("missing") or ()),
        rationale=exp.get("rationale", ""),
    )
    unknown_refs = [e for e in golden.evidence if e not in graph.nodes]
    if unknown_refs:
        raise CaseLoadError(
            f"{path}: expected.evidence references unknown node(s) {unknown_refs}"
        )

    return Case(
        case_id=case_id,
        title=raw.get("title", ""),
        tags=tuple(raw.get("tags") or ()),
        graph=graph,
        golden=golden,
        path=path,
    )


def load_cases(directory: str | Path) -> list[Case]:
    """Load every case in a directory, sorted by case id."""
    root = Path(directory)
    cases = [load_case(p) for p in sorted(root.glob("*.yaml"))]
    seen: dict[str, Path] = {}
    for case in cases:
        if case.case_id in seen:
            raise CaseLoadError(
                f"{case.path}: duplicate case_id {case.case_id!r} "
                f"(already defined in {seen[case.case_id]})"
            )
        seen[case.case_id] = case.path
    return sorted(cases, key=lambda c: c.case_id)
