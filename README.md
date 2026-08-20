# Provenance-Constrained Multimodal Investigation Agent

A prototype for the ArcellAI project brief: an agent that decides whether a
visual-inspection claim is supported by the evidence recorded for a case, cites
what it used, names what is missing or contradictory, and abstains or escalates
when the evidence does not reach a conclusion.

The technical plan is in **[PLAN.md](PLAN.md)** — the submission version, five
pages. `PLAN.html` is the same plan as a styled page with drawn diagrams and no
page limit, so it carries a little more detail. `demo.pptx` is a 14-slide walk
through it, and `demo-presenter-notes.pdf` prints those notes one page per
slide, expanded into a script. This file is how to run it.

## Run it

No API key and no network needed — the default models are deterministic stubs.

```bash
python3 -m venv .venv && ./.venv/bin/pip install pyyaml pytest
```

One case, end to end, with the full trace and an audit packet:

```bash
./.venv/bin/python run.py case case-010 --llm careful
```

The whole golden set, agent versus flattened-context baseline:

```bash
./.venv/bin/python run.py eval --per-case
```

The test suite, including the safety property the design exists to hold:

```bash
./.venv/bin/python -m pytest tests -q
```

Against a real model (`pip install anthropic`, `ANTHROPIC_API_KEY` set):

```bash
./.venv/bin/python run.py case case-011 --llm anthropic
./.venv/bin/python run.py eval --llm anthropic --baseline anthropic
```

## Layout

| Path | What it is |
|---|---|
| `pv/types.py` | The schema: nodes, edges, decisions, the conservatism lattice |
| `pv/graph.py` | Case graph, invariants, lineage traversal, content digest |
| `pv/loader.py` | Strict YAML fixture loader |
| `pv/policy.py` | Evidence admissibility — the one implementation both the tools and the validator use |
| `pv/tools.py` | The three agent tools and their JSON schemas |
| `pv/agent.py` | The tool-calling loop and its system prompt |
| `pv/validator.py` | The deterministic gate |
| `pv/audit.py` | The reproducible audit packet |
| `pv/baseline.py` | Flattened-context comparison system |
| `pv/evaluate.py` | Metrics and scoring |
| `pv/llm.py` | Stub, careful, adversarial, and Claude models |
| `cases/` | 17 hand-labelled fixture cases |
| `runs/` | Audit packets, one JSON file per run |

## Open questions

Three things the plan states in one line each, at length here — they are the
parts I would expect to be pressed on, and I would rather argue about them than
paper over them.

**`deciding_labels` does work a production system would have to earn.** The claim
spec declares which labels settle the question, and that is what makes polarity
deterministic rather than a model's opinion. It is honest for a bounded PPE
claim; it would not survive "was the site safe". The escalation path is to keep
the label vocabulary but let a reviewed, versioned LLM step author the spec per
claim *template* — never per request.

**Over-abstention is under-measured.** With 17 curated cases and stub reasoners,
the ceiling row reads 0.00, which is too flattering. The honest number needs a
real model over a larger, messier set. Over-abstention is the cost the constraint
imposes, and this fixture is not big enough to price it.

**`evidence_affecting` is an authoring judgement on the transformation**, not
something derived from what the operation did to the region a detection depends
on. A crop that removes the torso invalidates a vest detection but not a hat
detection; the prototype rejects both. Doing it properly needs region-aware
provenance — worth building, and worth being explicit that it is not built.

## Notes

- Hashes, URIs, sites, and detections in `cases/` are synthetic. The project is
  about how evidence governs an agent, not about detector accuracy.
- The stub models are naive reasoners used to make the harness runnable and
  reproducible offline. Their accuracy is not a claim about any real model's —
  see PLAN.md § Evaluation for what the comparison does and does not show.
- The Claude path is exercised in `tests/test_anthropic_path.py` against a
  local stand-in endpoint; it has not been run against the live API here.

## What is in `runs/`

Three sample audit packets and the evaluation results, checked in so a reviewer
can read the output without running anything:

| File | Shows |
|---|---|
| `case-010-*.json` | the decoy — two valid detections set aside for `SCOPE_MISMATCH`, downgraded to insufficient |
| `case-011-*.json` | an unresolved conflict, both sides preserved, escalated to human review |
| `case-013-*.json` | supersession resolving an apparent conflict against the confidence ordering |
| `eval.json` | every metric for all four systems, including the per-case silent-error list |

Regenerate them with `run.py case <id>` and `run.py eval --json runs/eval.json`.
