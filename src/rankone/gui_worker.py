"""Bounded child process used by the GUI; all computations reuse existing APIs."""

import json
import sys


def execute(request):
    from .operations import execute as run
    # Compatibility for the original demo's binary-only search requests.
    if isinstance(request, dict) and request.get("operation") == "search" and "q" not in request and "field" not in request:
        from .binary_membership import binary_membership_check
        from .binary_search import binary_maximum
        m, n = request.get("rows"), request.get("columns")
        if any(type(x) is not int or not 1 <= x <= 10 for x in (m,n)):
            raise ValueError("Use between 1 and 10 rows and columns.")
        report = binary_maximum(m,n,max_nodes=250000,max_candidates=100000,max_seconds=8)
        bundle = {"operation":"search","status":report["status"],"report":report}
        if report["lower_bound"]:
            checked = binary_membership_check(report["witness"])
            bundle["witness_check"] = checked
            if checked["status"] != "VALID":
                bundle["status"] = "INCONCLUSIVE"
        return bundle
    return run(request)


def main():
    try:
        raw = sys.stdin.read(2000001)
        if len(raw) > 2000000:
            raise ValueError("This demo accepts at most 2,000,000 input characters.")
        bundle = execute(json.loads(raw))
    except Exception as exc:
        bundle = {"status": "ERROR", "error": str(exc)}
    print(json.dumps(bundle, allow_nan=False))


if __name__ == "__main__":
    main()
