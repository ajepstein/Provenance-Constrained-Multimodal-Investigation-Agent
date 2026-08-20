"""Command line entry points: one walkthrough, or the whole evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pv.agent import run_case
from pv.audit import build_packet, write_packet
from pv.baseline import run_baseline
from pv.evaluate import (
    HEADER,
    answer_from_baseline,
    answer_from_run,
    corpus_index,
    score,
)
from pv.llm import build_model
from pv.loader import load_case, load_cases
from pv.policy import DEFAULT_POLICY

CASES_DIR = Path(__file__).resolve().parent.parent / "cases"


def cmd_case(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases)
    index = corpus_index(cases)
    case = next((c for c in cases if c.case_id == args.case_id), None)
    if case is None:
        print(f"no such case: {args.case_id}", file=sys.stderr)
        return 2

    model = build_model(args.llm)
    run = run_case(case, model, DEFAULT_POLICY, index)

    print(f"case      {case.case_id}  {case.title}")
    print(f"claim     {case.graph.claim.text.strip()}")
    print(f"graph     {len(case.graph.nodes)} nodes, {len(case.graph.edges)} edges, "
          f"digest {case.graph.digest()}")
    print(f"model     {run.model_name}")
    print()

    print("tool calls")
    for call in run.tool_calls:
        args_str = json.dumps(call.arguments, default=str)
        if len(args_str) > 88:
            args_str = args_str[:85] + "..."
        print(f"  {call.seq}. {call.name}{args_str}")
        print(f"     -> digest {call.result_digest}, "
              f"{len(call.result_ids)} ids referenced")
    print()

    decision = run.decision
    print(f"agent proposed   {decision.outcome if decision else '-'}")
    if decision:
        for assertion in decision.assertions:
            print(f"  - {assertion.statement}  [{', '.join(assertion.cited)}]")
        if decision.evidence_not_used:
            print("  set aside:")
            for unused in decision.evidence_not_used:
                print(f"    {unused.id}: {unused.reason}")
    print()

    print("validation")
    if not run.violations:
        print("  (no violations)")
    for violation in run.violations:
        print(f"  [{violation.severity:<9}] {violation.code}: {violation.message}")
    print()

    print(f"FINAL OUTCOME    {run.final_outcome}")
    print(f"expected (gold)  {case.golden.outcome}")
    print(f"match            {run.final_outcome == case.golden.outcome}")

    packet = build_packet(case, run, DEFAULT_POLICY)
    path = write_packet(packet, args.runs)
    print(f"\naudit packet     {path}  (content hash {packet['content_hash']})")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases)
    index = corpus_index(cases)
    systems = [s.strip() for s in args.llm.split(",") if s.strip()]
    rows = []
    detail: dict[str, dict[str, str]] = {}

    baseline_model = build_model(args.baseline)
    baseline_answers = {}
    for case in cases:
        result = run_baseline(case, baseline_model)
        baseline_answers[case.case_id] = answer_from_baseline(case, result)
    rows.append(score(f"baseline({args.baseline})", cases, baseline_answers))
    detail[f"baseline({args.baseline})"] = {
        c.case_id: baseline_answers[c.case_id].outcome for c in cases
    }

    for spec in systems:
        answers = {}
        for case in cases:
            model = build_model(spec)
            run = run_case(case, model, DEFAULT_POLICY, index)
            answers[case.case_id] = answer_from_run(run)
            if args.write_packets:
                write_packet(build_packet(case, run, DEFAULT_POLICY), args.runs)
        rows.append(score(f"agent({spec})", cases, answers))
        detail[f"agent({spec})"] = {c.case_id: answers[c.case_id].outcome for c in cases}

    print(f"{len(cases)} cases\n")
    print(HEADER)
    print("-" * len(HEADER))
    for row in rows:
        print(row.row())
    print()
    for row in rows:
        if row.silent_errors:
            print(f"{row.system}: silent errors on {', '.join(row.silent_errors)}")
        else:
            print(f"{row.system}: no silent errors")

    if args.per_case:
        print()
        names = list(detail)
        width = max(len(n) for n in names)
        print(f"{'case':<10} {'gold':<22} " + " ".join(f"{n:<22}" for n in names))
        for case in cases:
            cells = " ".join(f"{detail[n][case.case_id]:<22}" for n in names)
            print(f"{case.case_id:<10} {case.golden.outcome:<22} {cells}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {"cases": len(cases), "systems": [r.to_json() for r in rows]}, indent=2
            )
        )
        print(f"\nwrote {args.json}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pv", description=__doc__)
    parser.add_argument("--cases", default=str(CASES_DIR))
    parser.add_argument("--runs", default="runs")
    sub = parser.add_subparsers(dest="command", required=True)

    one = sub.add_parser("case", help="run one case end to end and print the trace")
    one.add_argument("case_id")
    one.add_argument("--llm", default="careful")
    one.set_defaults(func=cmd_case)

    ev = sub.add_parser("eval", help="score every system against the golden set")
    ev.add_argument("--llm", default="stub,careful,adversarial")
    ev.add_argument("--baseline", default="stub")
    ev.add_argument("--json", default="")
    ev.add_argument("--per-case", action="store_true")
    ev.add_argument("--write-packets", action="store_true")
    ev.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    return int(args.func(args))
