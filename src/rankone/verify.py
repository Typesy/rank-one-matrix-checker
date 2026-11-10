"""Replay mathematical evidence using library arithmetic and Gaussian ranks.

Does not call the optimized checker or its rank-one predicate. The finite-field
library remains a shared dependency, so this is not a formal proof kernel.
"""

from itertools import combinations, product

from .problem import prepare
from .reference import library_combine, library_rank, reference_check


class ReplayLimit(Exception):
    pass


def verify(report, *, max_directions=1000000, max_supports=100000):
    if isinstance(report,dict) and report.get('method')=='binary_ambient_membership':
        from .binary_membership import verify_binary_membership
        return verify_binary_membership(report,max_candidates=max_directions)
    if isinstance(report, dict) and report.get("method") == "annihilator-fibre":
        from .fibre import verify_fibre
        return verify_fibre(report, max_fibres=max_directions)
    for value in (max_directions, max_supports):
        if type(value) is not int or value < 0:
            raise ValueError("replay limits must be nonnegative integers")
    try:
        return _verify(report, max_directions, max_supports)
    except ReplayLimit:
        return {"verified": None, "reason": "replay_limit", "claim": report.get("status")}
    except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError, AttributeError):
        return {"verified": False, "reason": "malformed_or_false_certificate"}


def _verify(report, max_directions, max_supports):
    def answer(ok, reason):
        return {"verified": ok, "reason": reason, "claim": report.get("status")}

    if type(report["schema_version"]) is not int or report["schema_version"] != 1:
        return answer(False, "unknown_schema")
    f, matrices, _ = prepare(report["problem"])
    cert, k = report["certificate"], len(matrices)
    if report["status"] == "INCONCLUSIVE":
        return answer(None, "no_mathematical_decision_to_verify")
    if report["status"] not in ("VALID", "INVALID"):
        return answer(False, "unknown_status")
    if report["status"] == "INVALID":
        if report["reason"] == "generator_not_rank_one":
            i = cert["index"]
            if type(i) is not int or not 0 <= i < k:
                return answer(False, "invalid_index")
            r = library_rank(matrices[i], f)
            return answer(r != 1 and cert["rank"] == r, "generator_rank_replayed")
        coefficients = cert["coefficients"]
        if not isinstance(coefficients, list) or len(coefficients) != k:
            return answer(False, "invalid_coefficients")
        for value in coefficients:
            f.element(value)
        if not any(coefficients):
            return answer(False, "zero_coefficient_witness")
        matrix = library_combine(matrices, coefficients, f)
        if report["reason"] == "dependent_generators":
            return answer(library_rank(matrix, f) == 0, "dependence_replayed")
        if report["reason"] != "extra_rank_one":
            return answer(False, "unknown_invalid_reason")
        if any(library_rank(a, f) != 1 for a in matrices):
            return answer(False, "invalid_basis")
        if library_rank([sum(a, []) for a in matrices], f) != k:
            return answer(False, "invalid_basis")
        if sum(x != 0 for x in coefficients) < 2:
            return answer(False, "allowed_singleton_direction")
        if matrix != cert["matrix"] or library_rank(matrix, f) != 1:
            return answer(False, "false_rank_one_witness")
        if "u" in cert or "v" in cert:
            u, v = cert["u"], cert["v"]
            if len(u) != len(matrix) or len(v) != len(matrix[0]):
                return answer(False, "wrong_factor_sizes")
            for x in u + v:
                f.element(x)
            if (f.GF(u)[:, None] * f.GF(v)[None, :]).tolist() != matrix:
                return answer(False, "false_factorization")
        return answer(True, "extra_rank_one_replayed")

    if any(library_rank(a, f) != 1 for a in matrices):
        return answer(False, "invalid_generator")
    if library_rank([sum(a, []) for a in matrices], f) != k:
        return answer(False, "dependent_basis")
    if report["method"] == "reference-full":
        replay = reference_check(report["problem"], max_vectors=max_directions)
        if replay["status"] == "INCONCLUSIVE":
            raise ReplayLimit
        return answer(replay["status"] == "VALID" and replay["certificate"] == cert
                      and replay["stats"] == report["stats"], "full_reference_replayed")
    if report["method"] not in ("hybrid", "normalized"):
        return answer(False, "unknown_method")
    records = cert["supports"]
    if not isinstance(records, list):
        return answer(False, "invalid_support_records")
    if len(records) != 2 ** k - k - 1:
        return answer(False, "missing_or_duplicate_supports")
    if len(records) > max_supports:
        raise ReplayLimit
    by_support = {}
    for record in records:
        s = record["support"]
        if not isinstance(s, list) or any(type(i) is not int for i in s):
            return answer(False, "invalid_support")
        key = tuple(s)
        if not 2 <= len(s) <= k or sorted(set(s)) != s or any(i < 0 or i >= k for i in s):
            return answer(False, "invalid_support")
        if key in by_support:
            return answer(False, "duplicate_support")
        by_support[key] = record
    # Independently extract exact outer-product factors with library arithmetic.
    pairs = []
    for a in matrices:
        array = f.GF(a)
        r, c = next((r, c) for r, row in enumerate(a) for c, x in enumerate(row) if x)
        pairs.append((array[:, c].tolist(), (array[r, :] / array[r, c]).tolist()))
    checked, bounded = 0, 0
    for size in range(2, k + 1):
        for support in combinations(range(k), size):
            record = by_support[support]
            count = (f.q - 1) ** (size - 1)
            if type(record["directions"]) is not int or record["directions"] != count:
                return answer(False, "incorrect_coverage_count")
            if record["kind"] == "sylvester":
                ru = library_rank([pairs[i][0] for i in support], f)
                rv = library_rank([pairs[i][1] for i in support], f)
                if (record["rank_u"], record["rank_v"], record["lower_bound"]) != (ru, rv, ru + rv - size):
                    return answer(False, "false_support_bound")
                if ru + rv - size < 2:
                    return answer(False, "insufficient_support_bound")
                bounded += count
            elif record["kind"] == "enumerated":
                # Independent coefficient construction and full Gaussian rank.
                for tail in product(range(1, f.q), repeat=size - 1):
                    if checked >= max_directions:
                        raise ReplayLimit
                    coefficients = [0] * k
                    for i, x in zip(support, (1,) + tail):
                        coefficients[i] = x
                    if library_rank(library_combine(matrices, coefficients, f), f) < 2:
                        return answer(False, "mixed_combination_rank_below_two")
                    checked += 1
            else:
                return answer(False, "unknown_support_evidence")
    stats = report["stats"]
    if (stats["directions_checked"], stats["directions_by_bound"], stats["supports_completed"],
        stats["mixed_directions_total"]) != (checked, bounded, len(records), checked + bounded):
        return answer(False, "inconsistent_stats")
    if checked + bounded != (f.q ** k - 1) // (f.q - 1) - k:
        return answer(False, "incomplete_coverage")
    return answer(True, "all_support_evidence_replayed")
