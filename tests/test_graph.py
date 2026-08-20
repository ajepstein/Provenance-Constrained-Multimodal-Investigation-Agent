"""Structural invariants of the case graph."""

from __future__ import annotations

import pytest

from pv.graph import CaseGraph, GraphError
from pv.types import Artifact, Claim, Observation, Producer, Transformation


def _claim(case_id="c1"):
    return Claim(
        id="clm", case_id=case_id, text="t", predicate="p",
        deciding_labels=("hard_hat",), required_labels=("hard_hat",),
        requires=(), window_start="2026-01-01T00:00:00Z",
        window_end="2026-01-01T02:00:00Z", site="s",
    )


def _artifact(node_id="a1", case_id="c1", **kw):
    base = dict(
        media_type="image", uri="s3://x", sha256="abc",
        captured_at="2026-01-01T01:00:00Z", uploaded_by="op", site="s",
    )
    base.update(kw)
    return Artifact(id=node_id, case_id=case_id, **base)


def _observation(node_id="o1", case_id="c1", **kw):
    base = dict(
        obs_kind="detection", label="hard_hat", value="present",
        producer=Producer("model", "ppe-detector", "2.4.1"), confidence=0.9,
    )
    base.update(kw)
    return Observation(id=node_id, case_id=case_id, **base)


def test_duplicate_node_id_is_rejected():
    g = CaseGraph("c1")
    g.add_node(_artifact())
    with pytest.raises(GraphError, match="duplicate node id"):
        g.add_node(_artifact())


def test_node_from_another_case_is_rejected():
    g = CaseGraph("c1")
    with pytest.raises(GraphError, match="carries case_id"):
        g.add_node(_artifact(case_id="c2"))


def test_edge_to_unknown_node_is_rejected():
    g = CaseGraph("c1")
    g.add_node(_observation())
    with pytest.raises(GraphError, match="unknown node"):
        g.add_edge("o1", "derived_from", "missing")


def test_observation_without_a_source_is_rejected():
    g = CaseGraph("c1")
    g.add_node(_claim())
    g.add_node(_observation())
    with pytest.raises(GraphError, match="no derived_from edge"):
        g.seal()


def test_provenance_cycle_is_rejected():
    g = CaseGraph("c1")
    g.add_node(_claim())
    g.add_node(_artifact("a1"))
    g.add_node(_artifact("a2"))
    g.add_node(
        Transformation(id="t1", case_id="c1", op="crop", tool="x", tool_version="1")
    )
    g.add_edge("a1", "input_to", "t1")
    g.add_edge("t1", "produced", "a2")
    # a2 feeding the transformation that produced it closes the loop.
    g.add_edge("a2", "input_to", "t1")
    with pytest.raises(GraphError, match="provenance cycle"):
        g.seal()


def test_sealed_graph_refuses_new_evidence(by_id):
    graph = by_id["case-001"].graph
    with pytest.raises(GraphError, match="sealed"):
        graph.add_node(_artifact("new", case_id="case-001"))


def test_lineage_traverses_a_transformation_to_the_original(by_id):
    lineage = by_id["case-002"].graph.lineage("obs-002-a")
    assert lineage.roots == ("img-002-raw",)
    assert lineage.ops == ("resize@preproc:1.4.0",)
    assert lineage.complete


def test_lineage_reports_gaps_rather_than_guessing(by_id):
    lineage = by_id["case-017"].graph.lineage("obs-017-a")
    assert not lineage.complete
    assert any("hash" in gap for gap in lineage.gaps)
    assert any("uploader" in gap for gap in lineage.gaps)


def test_digest_is_stable_across_loads(by_id):
    from pv.loader import load_case

    first = by_id["case-001"].graph.digest()
    second = load_case(by_id["case-001"].path).graph.digest()
    assert first == second
