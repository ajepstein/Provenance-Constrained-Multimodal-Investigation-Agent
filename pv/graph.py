"""The case graph: nodes, typed edges, invariants, and lineage traversal.

Storage choice: plain dicts plus adjacency lists, serialisable to JSON. A
case here is <100 nodes and every traversal is an ancestor walk of depth <6,
so an embedded graph database or NetworkX would add a dependency and an
operational surface without answering a query we actually have. The upgrade
trigger is explicit: cross-case queries, concurrent writers, or a graph that
outlives one process should move this to SQLite (nodes/edges tables, an
append-only journal, and the same accessors).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Iterable

from pv.types import (
    Artifact,
    Claim,
    Edge,
    Node,
    Observation,
    Relation,
    Transformation,
)


class GraphError(ValueError):
    """An invariant was violated while building or using a case graph."""


@dataclass(frozen=True)
class LineageStep:
    node_id: str
    kind: str
    via: str | None  # the operation that produced this node, if any


@dataclass(frozen=True)
class Lineage:
    """The path from an item back to the original sources it rests on."""

    item_id: str
    roots: tuple[str, ...]
    steps: tuple[LineageStep, ...]
    ops: tuple[str, ...]
    evidence_affecting_ops: tuple[str, ...]
    gaps: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return bool(self.roots) and not self.gaps


class CaseGraph:
    """A single case: its claim, its evidence, and how the two connect.

    Two-phase lifecycle. Before `seal()` the graph is being built from source
    records. After `seal()` the evidence is frozen — the only permitted
    mutation is appending run nodes (tool calls, decisions), which keeps the
    audit trail in the same structure without letting a run rewrite the
    evidence it was judged against.
    """

    def __init__(self, case_id: str):
        self.case_id = case_id
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self._out: dict[str, list[Edge]] = defaultdict(list)
        self._in: dict[str, list[Edge]] = defaultdict(list)
        self._sealed = False
        self.run_nodes: dict[str, Any] = {}
        self.run_edges: list[Edge] = []

    # -- construction -----------------------------------------------------

    def add_node(self, node: Node) -> None:
        if self._sealed:
            raise GraphError(f"{self.case_id}: graph is sealed; cannot add {node.id!r}")
        if node.id in self.nodes:
            raise GraphError(f"{self.case_id}: duplicate node id {node.id!r}")
        if node.case_id != self.case_id:
            raise GraphError(
                f"node {node.id!r} carries case_id {node.case_id!r} "
                f"but was added to case {self.case_id!r}"
            )
        self.nodes[node.id] = node

    def add_edge(self, src: str, rel: Relation, dst: str, **meta: Any) -> None:
        if self._sealed:
            raise GraphError(f"{self.case_id}: graph is sealed; cannot add edges")
        for endpoint in (src, dst):
            if endpoint not in self.nodes:
                raise GraphError(
                    f"{self.case_id}: edge {src} -{rel}-> {dst} references "
                    f"unknown node {endpoint!r}"
                )
        edge = Edge(src=src, rel=rel, dst=dst, meta=meta)
        self.edges.append(edge)
        self._out[src].append(edge)
        self._in[dst].append(edge)

    def seal(self) -> "CaseGraph":
        """Check the structural invariants, then freeze the evidence."""
        self._check_invariants()
        self._sealed = True
        return self

    def _check_invariants(self) -> None:
        claims = [n for n in self.nodes.values() if isinstance(n, Claim)]
        if len(claims) != 1:
            raise GraphError(
                f"{self.case_id}: expected exactly one claim, found {len(claims)}"
            )

        # I1: no floating observations. Every observation must be *about*
        # something, or it is an assertion with no provenance at all.
        for node in self.nodes.values():
            if isinstance(node, Observation):
                if not self.out(node.id, "derived_from"):
                    raise GraphError(
                        f"{self.case_id}: observation {node.id!r} has no "
                        f"derived_from edge to a source artifact"
                    )

        # I2: derivation must be acyclic, or lineage never terminates.
        self._assert_acyclic()

    def _assert_acyclic(self) -> None:
        colour: dict[str, int] = {}

        def visit(node_id: str, path: list[str]) -> None:
            state = colour.get(node_id, 0)
            if state == 1:
                cycle = " -> ".join(path + [node_id])
                raise GraphError(f"{self.case_id}: provenance cycle: {cycle}")
            if state == 2:
                return
            colour[node_id] = 1
            for parent in self._parents(node_id):
                visit(parent, path + [node_id])
            colour[node_id] = 2

        for node_id in self.nodes:
            visit(node_id, [])

    # -- accessors --------------------------------------------------------

    def out(self, node_id: str, rel: Relation | None = None) -> list[Edge]:
        return [e for e in self._out[node_id] if rel is None or e.rel == rel]

    def into(self, node_id: str, rel: Relation | None = None) -> list[Edge]:
        return [e for e in self._in[node_id] if rel is None or e.rel == rel]

    @property
    def claim(self) -> Claim:
        return next(n for n in self.nodes.values() if isinstance(n, Claim))

    def observations(self) -> list[Observation]:
        return sorted(
            (n for n in self.nodes.values() if isinstance(n, Observation)),
            key=lambda n: n.id,
        )

    def artifacts(self) -> list[Artifact]:
        return sorted(
            (n for n in self.nodes.values() if isinstance(n, Artifact)),
            key=lambda n: n.id,
        )

    def get(self, node_id: str) -> Node | None:
        return self.nodes.get(node_id)

    def is_superseded(self, artifact_id: str) -> str | None:
        """Return the id of the artifact that replaced this one, if any."""
        for edge in self.into(artifact_id, "supersedes"):
            return edge.src
        return None

    def _parents(self, node_id: str) -> list[str]:
        """One step back along provenance: what this node was derived from."""
        node = self.nodes.get(node_id)
        parents: list[str] = []
        if isinstance(node, Observation):
            parents += [e.dst for e in self.out(node_id, "derived_from")]
        elif isinstance(node, Artifact):
            parents += [e.src for e in self.into(node_id, "produced")]
        elif isinstance(node, Transformation):
            parents += [e.src for e in self.into(node_id, "input_to")]
        return parents

    # -- traversal --------------------------------------------------------

    def lineage(self, item_id: str) -> Lineage:
        """Walk an item back to the original sources it ultimately rests on.

        Returns every intermediate step, the operations passed through, and
        an explicit list of *gaps* — points where provenance is unknown
        rather than clean. A gap is never silently treated as "fine".
        """
        if item_id not in self.nodes:
            raise GraphError(f"{self.case_id}: unknown item {item_id!r}")

        steps: list[LineageStep] = []
        ops: list[str] = []
        affecting: list[str] = []
        roots: list[str] = []
        gaps: list[str] = []
        seen: set[str] = set()

        def walk(node_id: str, via: str | None) -> None:
            if node_id in seen:
                return
            seen.add(node_id)
            node = self.nodes[node_id]
            steps.append(LineageStep(node_id, node.kind, via))

            if isinstance(node, Transformation):
                ops.append(f"{node.op}@{node.tool}:{node.tool_version}")
                if node.evidence_affecting:
                    affecting.append(f"{node.id}:{node.op}")

            if isinstance(node, Artifact):
                producers = self.into(node_id, "produced")
                if not producers:
                    roots.append(node_id)
                    if node.sha256 is None:
                        gaps.append(f"{node_id}: no content hash on the source file")
                    if node.uploaded_by is None:
                        gaps.append(f"{node_id}: no uploader recorded")
                    if node.captured_at is None:
                        gaps.append(f"{node_id}: capture time unknown")

            parents = self._parents(node_id)
            for parent in parents:
                parent_node = self.nodes[parent]
                label = (
                    f"{parent_node.op}"
                    if isinstance(parent_node, Transformation)
                    else None
                )
                walk(parent, label)

        walk(item_id, None)
        return Lineage(
            item_id=item_id,
            roots=tuple(sorted(roots)),
            steps=tuple(steps),
            ops=tuple(ops),
            evidence_affecting_ops=tuple(affecting),
            gaps=tuple(gaps),
        )

    def root_artifacts(self, item_id: str) -> list[Artifact]:
        return [self.nodes[r] for r in self.lineage(item_id).roots]  # type: ignore[misc]

    # -- run append (post-seal, append-only) ------------------------------

    def append_run_node(self, node_id: str, payload: dict[str, Any]) -> None:
        """Attach a tool call or decision record to the sealed case graph."""
        if node_id in self.run_nodes:
            raise GraphError(f"{self.case_id}: duplicate run node {node_id!r}")
        self.run_nodes[node_id] = payload

    def append_cites(self, decision_id: str, observation_id: str) -> None:
        self.run_edges.append(Edge(decision_id, "cites", observation_id))

    # -- identity ---------------------------------------------------------

    def digest(self) -> str:
        """Content hash of the sealed evidence.

        Two audit packets carrying the same digest were produced against
        byte-identical evidence, which is what makes a replay meaningful.
        """
        payload = {
            "case_id": self.case_id,
            "nodes": [_encode(self.nodes[k]) for k in sorted(self.nodes)],
            "edges": sorted(
                (e.src, e.rel, e.dst) for e in self.edges
            ),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "digest": self.digest(),
            "nodes": [_encode(self.nodes[k]) for k in sorted(self.nodes)],
            "edges": [
                {"src": e.src, "rel": e.rel, "dst": e.dst} for e in self.edges
            ],
        }


def _encode(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _encode(v) for k, v in asdict(value).items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    return value


def flatten_for_baseline(graph: CaseGraph) -> dict[str, Any]:
    """The same case, as an undifferentiated blob.

    This is what the baseline sees: every field is present, but nothing is
    typed as provenance, nothing is scoped, and no traversal is possible.
    The information is not withheld — only the structure is.
    """
    return {
        "claim": _encode(graph.claim),
        "records": [_encode(graph.nodes[k]) for k in sorted(graph.nodes)
                    if graph.nodes[k].kind != "claim"],
        "links": [f"{e.src} {e.rel} {e.dst}" for e in graph.edges],
    }
