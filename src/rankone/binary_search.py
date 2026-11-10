"""Bounded exact search for the largest good binary rank-one basis.

All rank-one factors are nonzero bit vectors. Distinct left/right factors are
necessary; increasing left factors give a unique order for every candidate set.
An independent good family has only good subfamilies, so failed prefixes prune
all their extensions. No equivalence-class reduction is assumed here.
"""

from collections import Counter
from time import monotonic


def outer_bits(u, v, m, n):
    """Row-major bits of uv^T; factor bit i is vector coordinate i."""
    return sum(v << (i * n) for i in range(m) if (u >> i) & 1)


def unpack_bits(value, m, n):
    return [[(value >> (i * n + j)) & 1 for j in range(n)] for i in range(m)]


def extend_span(span, generator, rank_one_set):
    """Return the new span, or a mathematical reason to reject the extension."""
    if generator in span:
        return None, "dependent"
    coset = tuple(x ^ generator for x in span)
    # The coset is disjoint from the old span. Its only allowed rank-one member
    # is the new generator itself (the term with x = 0).
    if any(x != generator and x in rank_one_set for x in coset):
        return None, "extra_rank_one"
    return span + coset, None


def binary_upper_bound(m, n):
    """Necessary distinct-factor and annihilator-counting bounds; not sharp generally.

    A k-dimensional good space has k rank-one directions. With r=mn-k linear
    constraints, each nonzero left factor admits at least 2**max(0,n-r)-1
    nonzero right factors. The transposed argument gives the second inequality.
    """
    if any(type(d) is not int or d < 1 for d in (m, n)):
        raise ValueError("dimensions must be positive integers")
    left, right = (1 << m) - 1, (1 << n) - 1
    initial = min(m * n, left, right)
    for k in range(initial, -1, -1):
        codimension = m * n - k
        lower_left = left * ((1 << max(0, n - codimension)) - 1)
        lower_right = right * ((1 << max(0, m - codimension)) - 1)
        if max(lower_left, lower_right) <= k:
            return k
    raise ArithmeticError("zero-dimensional space must satisfy the bounds")


def binary_maximum(m, n, *, max_nodes=1000000, max_candidates=100000,
                   max_span_size=1048576, max_seconds=None, method="quotient"):
    """Return EXACT with matching bounds, or INCONCLUSIVE with honest bounds.

    Budgets count attempted prefix extensions, not only successful families.
    A time limit is soft: checked between candidate/prefix operations. A completed
    exhaustive search is a computational upper bound; reaching the proved upper
    bound gives a witness plus a mathematical bound. The quotient method keeps
    one residual per ambient rank-one matrix, avoiding exponential span storage.
    max_span_size applies only to the independent reference search method 'span'.
    """
    if method not in ("span", "quotient"):
        raise ValueError("method must be span or quotient")
    for name, value in (("m", m), ("n", n)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (("max_nodes", max_nodes), ("max_candidates", max_candidates),
                        ("max_span_size", max_span_size)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if max_seconds is not None and (type(max_seconds) not in (int, float)
                                    or not 0 <= max_seconds < float("inf")):
        raise ValueError("max_seconds must be finite and nonnegative")
    started = monotonic()
    left_count, right_count = (1 << m) - 1, (1 << n) - 1
    upper = binary_upper_bound(m, n)
    candidate_count = left_count * right_count
    best = []
    stats = {"extensions_tested": 0, "dependent_prunes": 0,
             "extra_rank_one_prunes": 0, "valid_prefixes": 0,
             "size_bound_prunes": 0, "candidate_count": candidate_count}
    limits = {"max_nodes": max_nodes, "max_candidates": max_candidates,
              "max_span_size": max_span_size,
              "max_seconds": max_seconds}

    def finish(reason):
        exact = reason in ("proved_upper_bound_attained", "search_exhausted")
        return {"schema_version": 1, "method": "binary_partial_matchings_" + method,
                "field_order": 2, "rows": m, "columns": n,
                "status": "EXACT" if exact else "INCONCLUSIVE", "reason": reason,
                "lower_bound": len(best), "upper_bound": len(best) if exact else upper,
                "initial_upper_bound": upper,
                "upper_bound_justification": (
                    "complete_prefix_search" if reason == "search_exhausted"
                    else "dimension_distinct_factors_and_annihilator_counting"),
                "witness": {"field": {"order": 2}, "generators":
                            [unpack_bits(outer_bits(u, v, m, n), m, n) for u, v in best]},
                "factor_pairs_bits": [list(pair) for pair in best],
                "stats": {**stats, "elapsed_seconds": monotonic() - started},
                "limits": limits}

    def expired():
        return max_seconds is not None and monotonic() - started >= max_seconds

    if candidate_count > max_candidates:
        return finish("candidate_limit")
    candidates = {}
    for u in range(1, left_count + 1):
        for v in range(1, right_count + 1):
            if expired():
                return finish("time_limit")
            candidates[u, v] = outer_bits(u, v, m, n)
    rank_ones = frozenset(candidates.values())
    assert len(rank_ones) == candidate_count
    stop_reason = None

    def visit(pairs, state, last_left, used_right):
        nonlocal best, stop_reason
        if len(pairs) > len(best):
            best = list(pairs)
            if len(best) == upper:
                stop_reason = "proved_upper_bound_attained"
                return
        # Even selecting all remaining possible factors cannot improve best.
        available = min(left_count - last_left, right_count - len(used_right))
        if len(pairs) + available <= len(best):
            stats["size_bound_prunes"] += 1
            return
        frequencies = Counter(state.values()) if method == "quotient" else None
        for u in range(last_left + 1, left_count + 1):
            for v in range(1, right_count + 1):
                if v in used_right:
                    continue
                if expired():
                    stop_reason = "time_limit"
                    return
                if stats["extensions_tested"] >= max_nodes:
                    stop_reason = "node_limit"
                    return
                if method == "span" and 2 * len(state) > max_span_size:
                    stop_reason = "span_size_limit"
                    return
                stats["extensions_tested"] += 1
                if method == "span":
                    extended, reason = extend_span(state, candidates[u, v], rank_ones)
                else:
                    residual = state[u, v]
                    reason = ("dependent" if residual == 0 else
                              "extra_rank_one" if frequencies[residual] > 1 else None)
                    extended = None
                    if reason is None:
                        # Current residual map has kernel equal to the old span.
                        # The enlarged span includes rank-one R precisely when
                        # residual(R) is 0 or residual(A). Its latter fibre must
                        # consist of A alone. Eliminate one new pivot to update
                        # the quotient map, retaining every ambient candidate.
                        pivot = 1 << (residual.bit_length() - 1)
                        extended = {key: x ^ residual if x & pivot else x
                                    for key, x in state.items()}
                if reason is not None:
                    stats[reason + "_prunes"] += 1
                    continue
                stats["valid_prefixes"] += 1
                visit(pairs + [(u, v)], extended, u, used_right | {v})
                if stop_reason is not None:
                    return

    visit([], dict(candidates) if method == "quotient" else (0,), 0, frozenset())
    return finish(stop_reason or "search_exhausted")
