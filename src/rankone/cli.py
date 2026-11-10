"""Small local CLI: check, replay, and separately enumerate distributions."""

import argparse
import json
from pathlib import Path
import sys

from .checker import check, rank_distribution
from .binary_search import binary_maximum
from .binary_membership import binary_membership_check
from .reference import reference_check
from .verify import verify


def human_report(out):
    lines = [f"Decision: {out['status']}", f"Reason: {out['reason']}"]
    f = out["problem"]["field"]
    lines.append(f"Field: GF({f['order']}), characteristic {f['characteristic']}, degree {f['degree']}")
    if f["irreducible_poly"] is not None:
        lines.append(f"Defining polynomial coefficients (descending): {f['irreducible_poly']}")
    a = out["problem"]["generators"]
    m = len(a[0]) if a else out['problem']['rows']
    n = len(a[0][0]) if a else out['problem']['columns']
    lines.append(f"Input: {len(a)} matrices of size {m} x {n}")
    c, stats = out["certificate"], out["stats"]
    if out["status"] == "INVALID":
        if "coefficients" in c:
            lines.append(f"Counterexample coefficients: {c['coefficients']}")
        if "matrix" in c:
            lines.append(f"Resulting rank-one matrix: {c['matrix']}")
        if "u" in c:
            lines.append(f"Outer product factors: u={c['u']}, v={c['v']}")
        if "index" in c:
            lines.append(f"Generator index (zero-based): {c['index']}; rank: {c['rank']}")
    elif out["status"] == "VALID":
        if "fibres_checked" in stats:
            lines.append(f"Projective factor fibres checked: {stats['fibres_checked']} / {stats['fibres_total']}")
            lines.append(f"Rank-one directions in the span: {stats['rank_one_directions']}")
        elif "directions_checked" in stats:
            lines.append(f"Mixed directions evaluated: {stats['directions_checked']}; certified by bounds: {stats['directions_by_bound']}")
        else:
            lines.append(f"Mixed coefficient vectors evaluated: {stats['vectors_checked']}")
        lines.append("This certifies the supplied basis only. It gives a lower bound on the maximum dimension, not an upper bound or classification.")
    else:
        lines.append("No mathematical decision: the requested computation did not finish.")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("check", "verify", "distribution", "binary-membership"):
        p = sub.add_parser(command)
        p.add_argument("input", type=Path)
        p.add_argument("--output", type=Path)
        p.add_argument("--max-directions", type=int, default=1000000)
        if command == "binary-membership":
            p.add_argument("--max-candidates", type=int, default=100000)
        if command == 'distribution':
            p.add_argument('--method',choices=('auto','enumerate'),default='auto')
            p.add_argument('--max-seconds',type=float)
        if command in ("check", "verify"):
            p.add_argument("--max-supports", type=int, default=100000)
        if command == "check":
            p.add_argument("--method", choices=("fibre", "hybrid", "normalized", "reference"), default="hybrid")
            p.add_argument("--max-seconds", type=float)
            p.add_argument("--text", action="store_true")
    p = sub.add_parser("binary-maximum")
    p.add_argument("rows", type=int)
    p.add_argument("columns", type=int)
    p.add_argument("--output", type=Path)
    p.add_argument("--max-nodes", type=int, default=1000000)
    p.add_argument("--max-candidates", type=int, default=100000)
    p.add_argument("--max-span-size", type=int, default=1048576)
    p.add_argument("--method", choices=("quotient", "span"), default="quotient")
    p.add_argument("--max-seconds", type=float)
    research_commands = ("bounds", "table", "construct", "search", "classes", "enumerate-classes", "equivalent", "canonical", "classify", "replay", "factor-analysis")
    for command in research_commands:
        p = sub.add_parser(command)
        p.add_argument("--output", type=Path)
        p.add_argument("--text", action="store_true")
        p.add_argument("--max-seconds", type=float, default=15)
        p.add_argument("--max-candidates", type=int, default=100000)
        if command in ("bounds", "construct", "search", "classes", "enumerate-classes", "table"):
            p.add_argument("q", type=int)
            if command != "table":
                p.add_argument("rows", type=int)
                p.add_argument("columns", type=int)
            p.add_argument("--polynomial", type=int, nargs="+")
        if command == "construct":
            p.add_argument("--kind", choices=("verified", "two_row", "vandermonde"), default="verified")
            p.add_argument("--size", type=int)
            p.add_argument("--include-infinity", action="store_true")
            p.add_argument("--variant", choices=("circuit", "independent"), default="circuit")
        if command == "search":
            p.add_argument("--method", choices=("greedy", "exhaustive"), default="greedy")
            p.add_argument("--trials", type=int, default=4)
            p.add_argument("--seed", type=int, default=20261026)
            p.add_argument("--max-nodes", type=int, default=50000)
            p.add_argument("--fresh", action="store_true", help="Search without starting from a stored witness")
        if command in ("classes","enumerate-classes"):
            p.add_argument("size", type=int)
        if command in ("equivalent", "canonical", "classify", "classes","enumerate-classes"):
            p.add_argument("--transpose", action="store_true")
            p.add_argument("--frobenius", action="store_true")
        if command == "equivalent":
            p.add_argument("input_a", type=Path)
            p.add_argument("input_b", type=Path)
        elif command in ("canonical", "replay","factor-analysis"):
            p.add_argument("input", type=Path)
        elif command == "classify":
            p.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args(argv)
    try:
        if args.command in research_commands:
            from .operations import execute
            from .presentation import research_explanation
            from .demo import unwrap_problem
            request = {"operation": {"equivalent": "equivalence", "classes": "known_classes", "enumerate-classes":"enumerate_classes", "factor-analysis":"factor_analysis"}.get(args.command, args.command),
                       "max_seconds": args.max_seconds, "max_candidates": args.max_candidates}
            for attr in ("q", "rows", "columns", "method", "trials", "seed", "max_nodes", "variant", "include_infinity"):
                if hasattr(args, attr): request[attr] = getattr(args, attr)
            if hasattr(args, "polynomial") and args.polynomial is not None:
                request["field"] = {"order": args.q, "irreducible_poly": args.polynomial}
            if hasattr(args, "transpose"):
                request.update(allow_transpose=args.transpose, allow_frobenius=args.frobenius)
            if args.command == "construct": request.update(construction=args.kind, k=args.size)
            if args.command in ("classes","enumerate-classes"): request["k"] = args.size
            if args.command == "search": request["use_known"] = not args.fresh
            if args.command == "equivalent":
                request.update(problem_a=unwrap_problem(json.loads(args.input_a.read_text())),
                               problem_b=unwrap_problem(json.loads(args.input_b.read_text())))
            if args.command in ("canonical","factor-analysis"): request["problem"] = unwrap_problem(json.loads(args.input.read_text()))
            if args.command == "replay": request["data"] = json.loads(args.input.read_text())
            if args.command == "classify": request["problems"] = [unwrap_problem(json.loads(p.read_text())) for p in args.inputs]
            bundle = execute(request)
            if args.output: args.output.write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n")
            print(research_explanation(bundle) if args.text else json.dumps(bundle, indent=2, allow_nan=False))
            if bundle['status'] in ('INVALID', 'INVALID_INPUT', 'REJECTED'): return 1
            if bundle['status'] in ('INCONCLUSIVE', 'BOUNDED', 'PARTIAL_CLASSIFICATION', 'UNAVAILABLE'): return 2
            return 0
        data = None if args.command == "binary-maximum" else json.loads(args.input.read_text())
        if args.command == "binary-maximum":
            out = binary_maximum(args.rows, args.columns, max_nodes=args.max_nodes,
                                 max_candidates=args.max_candidates,
                                 max_span_size=args.max_span_size,
                                 method=args.method,
                                 max_seconds=args.max_seconds)
            code = 0 if out["status"] == "EXACT" else 2
        elif args.command == "binary-membership":
            out = binary_membership_check(data, max_candidates=args.max_candidates)
            code = {"VALID": 0, "INVALID": 1, "INCONCLUSIVE": 2}[out["status"]]
        elif args.command == "check":
            if args.method == "reference":
                if args.max_seconds is not None:
                    raise ValueError("reference mode has a vector limit only; omit --max-seconds")
                out = reference_check(data, max_vectors=args.max_directions)
            else:
                out = check(data, method=args.method, max_directions=args.max_directions,
                            max_supports=args.max_supports, max_seconds=args.max_seconds)
            code = {"VALID": 0, "INVALID": 1, "INCONCLUSIVE": 2}[out["status"]]
        elif args.command == "verify":
            out = verify(data, max_directions=args.max_directions, max_supports=args.max_supports)
            code = 0 if out["verified"] is True else 2 if out["verified"] is None else 1
        else:
            out = rank_distribution(data, max_directions=args.max_directions,method=args.method,max_seconds=args.max_seconds)
            code = 0 if out["status"] == "COMPLETE" else 2
        encoded = json.dumps(out, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(encoded)
        print(human_report(out) if args.command == "check" and args.text else encoded, end="\n" if args.command == "check" and args.text else "")
        return code
    except (ValueError, TypeError, OSError) as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
