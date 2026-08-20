"""Exercise the real Claude code path against a local stand-in endpoint.

There is no way to unit-test a model's judgement, but the plumbing around it -
request serialisation, tool_use parsing, echoing the assistant turn back with
its thinking blocks intact, and the tool_result round trip - is ordinary code
and is tested here. The server below speaks the Messages API shape; the client
is the real SDK pointed at it.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from pv.agent import run_case
from pv.llm import AnthropicModel

pytest.importorskip("anthropic")

_TURNS = [
    {
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {
                "type": "tool_use",
                "id": "tu_1",
                "name": "retrieve_evidence",
                "input": {"question": "does the evidence support the claim?"},
            },
        ],
        "stop_reason": "tool_use",
    },
    {
        "content": [
            {
                "type": "tool_use",
                "id": "tu_2",
                "name": "record_decision",
                "input": {
                    "outcome": "needs_human_review",
                    "assertions": [
                        {"statement": "hard_hat is present", "cited": ["obs-011-a"]},
                        {"statement": "hard_hat is absent", "cited": ["obs-011-c"]},
                    ],
                    "evidence_used": ["obs-011-a", "obs-011-b", "obs-011-c"],
                    "evidence_not_used": [],
                    "summary": "Admissible sources disagree about hard_hat.",
                },
            }
        ],
        "stop_reason": "tool_use",
    },
    {
        "content": [{"type": "text", "text": "Escalated to review."}],
        "stop_reason": "end_turn",
    },
]


class _Handler(BaseHTTPRequestHandler):
    turn = 0
    requests: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        length = int(self.headers.get("content-length", 0))
        _Handler.requests.append(json.loads(self.rfile.read(length)))
        turn = _TURNS[min(_Handler.turn, len(_TURNS) - 1)]
        _Handler.turn += 1
        body = json.dumps(
            {
                "id": f"msg_{_Handler.turn}",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5",
                "content": turn["content"],
                "stop_reason": turn["stop_reason"],
                "stop_sequence": None,
                "usage": {"input_tokens": 100, "output_tokens": 50},
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # keep pytest output clean
        return


@pytest.fixture
def fake_api(monkeypatch):
    _Handler.turn = 0
    _Handler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv(
        "ANTHROPIC_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}"
    )
    yield _Handler
    server.shutdown()


def test_claude_loop_drives_a_case_to_a_validated_outcome(by_id, fake_api):
    case = by_id["case-011"]
    run = run_case(case, AnthropicModel("claude-opus-5"))

    assert [c.name for c in run.tool_calls] == [
        "retrieve_evidence",
        "record_decision",
    ]
    assert run.decision.outcome == "needs_human_review"
    assert run.final_outcome == "needs_human_review"

    sent = fake_api.requests
    first = sent[0]
    assert first["model"] == "claude-opus-5"
    assert first["thinking"] == {"type": "adaptive"}
    assert {t["name"] for t in first["tools"]} == {
        "retrieve_evidence",
        "inspect_provenance",
        "record_decision",
    }
    # Every tool must ship a usable schema, or the model cannot call it.
    for tool in first["tools"]:
        assert tool["input_schema"]["type"] == "object"
        assert tool["description"]

    # The assistant turn is echoed back unchanged - thinking blocks included -
    # and the tool result is returned in a single user message keyed by id.
    second = sent[1]
    assistant = second["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant["content"][0]["type"] == "thinking"
    tool_result = second["messages"][2]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "tu_1"
    payload = json.loads(tool_result["content"])
    assert payload["case_id"] == "case-011"
    assert payload["unresolved_conflicts"][0]["label"] == "hard_hat"
