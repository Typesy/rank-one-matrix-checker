"""Complete normalized checking, optionally certified by support rank bounds."""

from itertools import combinations, product
from time import monotonic

from .linalg import combine, elimination, factors, flatten, rank, rank_one
from .problem import prepare, result


def supports(k, minimum=2):
    for size in range(minimum, k + 1):
        yield from combinations(range(k), size)


def normalized_coefficients(k, support, q):
    for tail in product(range(1, q), repeat=len(support) - 1):
        coefficients = [0] * k
        coefficients[support[0]] = 1
        for index, value in zip(support[1:], tail):
            coefficients[index] = value
        yield coefficients


def check(problem, *, method="hybrid", max_directions=1000000,
          max_supports=100000, max_seconds=None):
    if method in ("fibre", "annihilator-fibre"):
        from .fibre import fibre_check
        return fibre_check(problem, max_fibres=max_directions, max_seconds=max_seconds)
    if method not in ("hybrid", "normalized"):
        raise ValueError("method must be fibre, hybrid or normalized")
    for name, value in (("max_directions", max_directions), ("max_supports", max_supports)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if max_seconds is not None and (type(max_seconds) not in (int, float)
                                    or not 0 <= max_seconds < float("inf")):
        raise ValueError("max_seconds must be finite and nonnegative")
    started = monotonic()
    f, matrices, canonical = prepare(problem)
    k = len(matrices)
    stats = {"directions_checked": 0, "directions_by_bound": 0,
             "supports_completed": 0,
             "mixed_directions_total": (f.q ** k - 1) // (f.q - 1) - k}
    records = []

    def finish(status, reason, certificate):
        return result(canonical, method, status, reason, certificate,
                      {**stats, "elapsed_seconds": monotonic() - started,
                       "limits": {"max_directions": max_directions,
                                  "max_supports": max_supports,
                                  "max_seconds": max_seconds}})

    def expired():
        return max_seconds is not None and monotonic() - started >= max_seconds

    for i, matrix in enumerate(matrices):
        if expired():
            return finish("INCONCLUSIVE", "time_limit", {"supports": records})
        if not rank_one(matrix, f):
            return finish("INVALID", "generator_not_rank_one", {"index": i, "rank": rank(matrix, f)})
    dimension, relation = elimination([flatten(a) for a in matrices], f, track=True)
    if dimension != k:
        return finish("INVALID", "dependent_generators", {"coefficients": relation})
    factor_pairs = [factors(matrix, f) for matrix in matrices]
    for support in supports(k):
        if expired():
            return finish("INCONCLUSIVE", "time_limit", {"supports": records})
        if stats["supports_completed"] >= max_supports:
            return finish("INCONCLUSIVE", "support_limit", {"supports": records})
        count = (f.q - 1) ** (len(support) - 1)
        if method == "hybrid":
            # Rank of a list of factors equals rank of its transpose.
            ru = rank([factor_pairs[i][0] for i in support], f)
            rv = rank([factor_pairs[i][1] for i in support], f)
            lower = ru + rv - len(support)
            if lower >= 2:
                records.append({"support": list(support), "kind": "sylvester",
                                "rank_u": ru, "rank_v": rv,
                                "lower_bound": lower, "directions": count})
                stats["supports_completed"] += 1
                stats["directions_by_bound"] += count
                continue
        for coefficients in normalized_coefficients(k, support, f.q):
            if expired():
                return finish("INCONCLUSIVE", "time_limit", {"supports": records})
            if stats["directions_checked"] >= max_directions:
                return finish("INCONCLUSIVE", "direction_limit", {"supports": records})
            matrix = combine(matrices, coefficients, f)
            stats["directions_checked"] += 1
            if not any(flatten(matrix)):
                raise ArithmeticError("nonzero coefficients vanished after independence check")
            if rank_one(matrix, f):
                u, v = factors(matrix, f)
                return finish("INVALID", "extra_rank_one", {
                    "coefficients": coefficients, "matrix": matrix, "u": u, "v": v})
        records.append({"support": list(support), "kind": "enumerated", "directions": count})
        stats["supports_completed"] += 1
    return finish("VALID", "all_mixed_supports_certified", {"supports": records})


def rank_distribution(problem, *, max_directions=1000000, method="enumerate", max_seconds=None):
    """Separate complete enumeration; never invent counts from a lower bound."""
    if type(max_directions) is not int or max_directions < 0:
        raise ValueError("max_directions must be a nonnegative integer")
    if method not in ('enumerate','auto'):
        raise ValueError("distribution method must be enumerate or auto")
    if max_seconds is not None and (type(max_seconds) not in (int,float) or not 0<=max_seconds<float('inf')):
        raise ValueError("max_seconds must be finite and nonnegative")
    started=monotonic()
    f, matrices, canonical = prepare(problem)
    k = len(matrices)
    if rank([flatten(a) for a in matrices], f) != k:
        raise ValueError("rank distribution requires independent input generators")
    if method=='auto' and all(rank_one(a,f) for a in matrices):
        pairs=[factors(a,f) for a in matrices]
        if min(rank([u for u,v in pairs],f),rank([v for u,v in pairs],f))<=2:
            from .analysis import two_row_distribution
            out=two_row_distribution(canonical,max_fibres=max_directions,max_seconds=max_seconds)
            if out['status'] in ('COMPLETE','INCONCLUSIVE'):
                return out
    counts, checked = {0: 1}, 0
    for support in supports(k, minimum=1):
        for coefficients in normalized_coefficients(k, support, f.q):
            expired=max_seconds is not None and monotonic()-started>=max_seconds
            if checked >= max_directions or expired:
                return {"status": "INCONCLUSIVE", "reason": "time_limit" if expired else "direction_limit",
                        "directions_checked": checked,
                        "partial_matrix_counts": {str(r): v for r, v in counts.items()},
                        "problem": canonical}
            r = rank(combine(matrices, coefficients, f), f)
            counts[r] = counts.get(r, 0) + f.q - 1
            checked += 1
    return {"status": "COMPLETE", "matrix_counts": {str(r): v for r, v in sorted(counts.items())},
            "total_matrices": f.q ** k, "directions_checked": checked, "problem": canonical}
