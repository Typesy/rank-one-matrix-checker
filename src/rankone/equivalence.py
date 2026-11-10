"""Exact, bounded equivalence of good rank-one matrix spaces.

The paired projective factors retain one common generator permutation. A
completed basis-image search proves a negative result; interrupted searches
never do. Positive transporters are replayed directly with library arithmetic.
"""

from itertools import permutations, product
from copy import deepcopy
import json
from math import isfinite
from time import monotonic

from .linalg import factors, flatten, rank, rank_one
from .problem import prepare


class _Limit(Exception):
    pass


class _Invalid(Exception):
    pass


class _Budget:
    def __init__(self, candidates, seconds):
        if type(candidates) is not int or candidates < 0:
            raise ValueError("max_candidates must be a nonnegative integer")
        if seconds is not None and (type(seconds) not in (int, float)
                                    or not isfinite(seconds) or seconds < 0):
            raise ValueError("max_seconds must be finite and nonnegative")
        self.maximum, self.seconds = candidates, seconds
        self.started, self.count = monotonic(), 0

    def remaining(self):
        if self.seconds is None:
            return None
        return max(0, self.seconds - (monotonic() - self.started))

    def tick(self):
        if self.count >= self.maximum or self.remaining() == 0:
            raise _Limit
        self.count += 1

    def stats(self):
        return {"candidates_checked": self.count,
                "elapsed_seconds": monotonic() - self.started,
                "max_candidates": self.maximum, "max_seconds": self.seconds}


def _shape(problem):
    a = problem["generators"]
    return (len(a[0]), len(a[0][0])) if a else (problem["rows"], problem["columns"])


def _action(transpose, frobenius):
    if type(transpose) is not bool or type(frobenius) is not bool:
        raise ValueError("equivalence options must be booleans")
    return {"left_right_linear": True, "allow_transpose": transpose,
            "allow_frobenius": frobenius,
            "generator_order": "unordered", "generator_scaling": "nonzero"}


def _identity(n):
    return [[int(i == j) for j in range(n)] for i in range(n)]


def _transpose(a):
    return [list(row) for row in zip(*a)]


def _matmul(a, b, f):
    return [[_dot(row, col, f) for col in zip(*b)] for row in a]


def _dot(a, b, f):
    out = 0
    for x, y in zip(a, b):
        out = f.add(out, f.mul(x, y))
    return out


def _matvec(a, v, f):
    return [_dot(row, v, f) for row in a]


def _inverse(a, f):
    n = len(a)
    work = [list(row) + eye for row, eye in zip(a, _identity(n))]
    for j in range(n):
        pivot = next((i for i in range(j, n) if work[i][j]), None)
        if pivot is None:
            raise ValueError("singular matrix")
        work[j], work[pivot] = work[pivot], work[j]
        c = f.inv(work[j][j])
        work[j] = [f.mul(c, x) for x in work[j]]
        for i in range(n):
            if i != j and work[i][j]:
                c = work[i][j]
                work[i] = [f.sub(x, f.mul(c, y)) for x, y in zip(work[i], work[j])]
    return [row[n:] for row in work]


def _columns(vectors):
    return [list(row) for row in zip(*vectors)]


def _normalize(v, f):
    pivot = next(x for x in v if x)
    scale = f.inv(pivot)
    return tuple(f.mul(scale, x) for x in v)


def _basis_indices(vectors, f):
    chosen, rows = [], []
    for i, v in enumerate(vectors):
        if rank(rows + [v], f) > len(rows):
            chosen.append(i)
            rows.append(v)
    return chosen


def _compress(points, dimension, f):
    """Coordinates in an intrinsic support, with a full ambient change map."""
    indices = _basis_indices(points, f)
    basis = [list(points[i]) for i in indices]
    r = len(basis)
    for e in _identity(dimension):
        if rank(basis + [e], f) > len(basis):
            basis.append(e)
    inverse = _inverse(_columns(basis), f)
    coordinates = [list(_normalize(_matvec(inverse, v, f)[:r], f)) for v in points]
    return coordinates, inverse, r


def _padded_map(small, ambient, support_inverse, f):
    out = _identity(ambient)
    r = len(small)
    for i in range(r):
        out[i] = list(small[i]) + [0] * (ambient-r)
    return _matmul(out, support_inverse, f)


def _power(value, exponent, f):
    out = 1
    while exponent:
        if exponent & 1:
            out = f.mul(out, value)
        value = f.mul(value, value)
        exponent //= 2
    return out


def _variant(matrices, f, transposed, exponent):
    out = [[[ _power(x, f.p**exponent, f) for x in row] for row in a]
           for a in matrices]
    return [_transpose(a) for a in out] if transposed else out


def _variants(matrices, f, allow_transpose, allow_frobenius, square):
    for exponent in range(f.degree if allow_frobenius else 1):
        for transposed in range(2 if allow_transpose and square else 1):
            yield _variant(matrices, f, bool(transposed), exponent), bool(transposed), exponent


def _factor_data(matrices, m, n, f):
    pairs = [factors(a, f) for a in matrices]
    left = [list(_normalize(u, f)) for u, _ in pairs]
    right = [list(_normalize(v, f)) for _, v in pairs]
    u, ei, r = _compress(left, m, f)
    v, fi, s = _compress(right, n, f)
    return u, v, ei, fi, r, s


def _validate(canonical, budget):
    """Input goodness is a prerequisite for the intrinsic-basis reduction."""
    if not canonical["generators"]:
        return {"status": "VALID", "reason": "zero_space"}
    if budget.remaining() == 0:
        raise _Limit
    try:
        from .fibre import fibre_check
        checked = fibre_check(canonical, max_fibres=budget.maximum,
                              max_seconds=budget.remaining())
    except ImportError:
        from .checker import check
        checked = check(canonical, max_directions=budget.maximum,
                        max_supports=budget.maximum, max_seconds=budget.remaining())
    if checked["status"] == "INCONCLUSIVE":
        raise _Limit
    if checked["status"] != "VALID":
        raise _Invalid(checked["reason"])
    return {"status": "VALID", "reason": checked["reason"]}


def _right_map(v, target, permutation, f):
    """Solve the diagonal ratios after fixing matched right-basis images."""
    s = len(v[0])
    basis = _basis_indices(v, f)
    images = [target[permutation[i]] for i in basis]
    if rank(images, f) != s:
        return None
    vi = _inverse(_columns([v[i] for i in basis]), f)
    yi = _inverse(_columns(images), f)
    c = [_matvec(vi, x, f) for x in v]
    d = [_matvec(yi, target[permutation[i]], f) for i in range(len(v))]
    graph = [[] for _ in range(s)]
    for x, y in zip(c, d):
        support = [j for j, z in enumerate(x) if z]
        if support != [j for j, z in enumerate(y) if z]:
            return None
        first = support[0]
        for j in support[1:]:
            # beta_j / beta_first = (d_j/c_j)/(d_first/c_first)
            ratio = f.mul(f.mul(y[j], f.inv(x[j])),
                          f.inv(f.mul(y[first], f.inv(x[first]))))
            graph[first].append((j, ratio))
            graph[j].append((first, f.inv(ratio)))
    beta = [None] * s
    for root in range(s):
        if beta[root] is not None:
            continue
        beta[root], stack = 1, [root]
        while stack:
            i = stack.pop()
            for j, ratio in graph[i]:
                value = f.mul(beta[i], ratio)
                if beta[j] is None:
                    beta[j] = value
                    stack.append(j)
                elif beta[j] != value:
                    return None
    images_scaled = [[f.mul(x, b) for x in col] for col, b in zip(images, beta)]
    transform = _matmul(_columns(images_scaled), vi, f)
    if any(_normalize(_matvec(transform, x, f), f) != tuple(target[permutation[i]])
           for i, x in enumerate(v)):
        raise ArithmeticError("right ratio solver produced a false map")
    return transform


def _transporter(u, v, x, y, f, budget):
    r = len(u[0])
    basis = _basis_indices(u, f)
    inverse = _inverse(_columns([u[i] for i in basis]), f)
    lookup = {tuple(point): i for i, point in enumerate(x)}
    for image_indices in permutations(range(len(x)), r):
        images = [x[i] for i in image_indices]
        if rank(images, f) != r:
            continue
        for tail in product(range(1, f.q), repeat=r-1):
            budget.tick()
            scales = (1,) + tail
            scaled = [[f.mul(z, c) for z in col] for col, c in zip(images, scales)]
            transform = _matmul(_columns(scaled), inverse, f)
            image_points = [_normalize(_matvec(transform, point, f), f) for point in u]
            if any(point not in lookup for point in image_points):
                continue
            permutation = [lookup[point] for point in image_points]
            right = _right_map(v, y, permutation, f)
            if right is not None:
                return transform, right, permutation
    return None


def _certificate(source_variant, target, p, q, permutation, transposed, exponent, f):
    scalars = []
    for a, j in zip(source_variant, permutation):
        image = _matmul(_matmul(p, a, f), q, f)
        b = target[j]
        row, col = next((i, l) for i, row in enumerate(b) for l, z in enumerate(row) if z)
        scalar = f.mul(image[row][col], f.inv(b[row][col]))
        if not scalar or image != [[f.mul(scalar, z) for z in row] for row in b]:
            raise ArithmeticError("false equivalence transporter")
        scalars.append(scalar)
    return {"P": p, "Q": q, "permutation": permutation, "scalars": scalars,
            "transpose": transposed, "frobenius_power": exponent}


def equivalent(problem_a, problem_b, *, allow_transpose=False, allow_frobenius=False,
               max_candidates=100000, max_seconds=None):
    """Decide the declared equivalence, with complete negatives or a limit."""
    budget = _Budget(max_candidates, max_seconds)
    action = _action(allow_transpose, allow_frobenius)
    f, a, source = prepare(problem_a)
    g, b, target = prepare(problem_b)
    if f.metadata() != g.metadata():
        raise ValueError("equivalence requires the same explicit field model")
    m, n = _shape(source)
    out = {"schema_version": 1, "method": "paired_projective_transporters",
           "source": source, "target": target, "action": action}

    def finish(status, reason, certificate=None):
        return {**out, "status": status, "reason": reason,
                "certificate": certificate or {}, "stats": budget.stats()}

    try:
        out["input_checks"] = [_validate(source, budget), _validate(target, budget)]
        if (m, n) != _shape(target):
            return finish("INEQUIVALENT", "ambient_shapes_differ")
        if len(a) != len(b):
            return finish("INEQUIVALENT", "dimensions_differ")
        if not a:
            return finish("EQUIVALENT", "zero_spaces", {
                "P": _identity(m), "Q": _identity(n), "permutation": [],
                "scalars": [], "transpose": False, "frobenius_power": 0})
        x, y, ex, fy, rx, sy = _factor_data(b, m, n, f)
        seen_ranks = []
        for variant, transposed, exponent in _variants(a, f, allow_transpose,
                                                       allow_frobenius, m == n):
            u, v, eu, fv, r, s = _factor_data(variant, m, n, f)
            seen_ranks.append([r, s])
            if (r, s) != (rx, sy):
                continue
            # Enumerate the smaller side; swapping factors preserves the matching.
            if (len(a) ** s) * (f.q-1)**max(0, s-1) < (len(a) ** r) * (f.q-1)**max(0, r-1):
                match = _transporter(v, u, y, x, f, budget)
                if match:
                    right, left, permutation = match
                else:
                    continue
            else:
                match = _transporter(u, v, x, y, f, budget)
                if match:
                    left, right, permutation = match
                else:
                    continue
            # Coordinates target -> ambient use inverse of its compression map.
            p0 = _padded_map(left, m, eu, f)
            r0 = _padded_map(right, n, fv, f)
            p = _matmul(_inverse(ex, f), p0, f)
            right_full = _matmul(_inverse(fy, f), r0, f)
            cert = _certificate(variant, b, p, _transpose(right_full), permutation,
                                transposed, exponent, f)
            return finish("EQUIVALENT", "transporter_found", cert)
        if all(tuple(rs) != (rx, sy) for rs in seen_ranks):
            return finish("INEQUIVALENT", "essential_support_ranks_differ", {
                "source_variant_ranks": seen_ranks, "target_ranks": [rx, sy]})
        return finish("INEQUIVALENT", "complete_transporter_search", {
            "exhausted": True, "source_variant_ranks": seen_ranks,
            "target_ranks": [rx, sy], "candidates_checked": budget.count})
    except _Limit:
        return finish("INCONCLUSIVE", "work_or_time_limit")
    except _Invalid as exc:
        return finish("INVALID_INPUT", "input_is_not_a_good_basis", {"input_reason": str(exc)})


def verify_equivalence(report, *, max_candidates=100000, max_seconds=None):
    """Replay a positive map directly; replay complete negatives within limits."""
    try:
        if type(report.get("schema_version")) is not int or report["schema_version"] != 1:
            return {"verified": False, "reason": "unknown_schema"}
        if report["status"] in ("INCONCLUSIVE", "INVALID_INPUT"):
            return {"verified": None, "reason": "no_equivalence_claim"}
        f, a, source = prepare(report["source"])
        g, b, target = prepare(report["target"])
        if f.metadata() != g.metadata():
            return {"verified": False, "reason": "field_model_mismatch"}
        action = report["action"]
        if action != _action(action["allow_transpose"], action["allow_frobenius"]):
            return {"verified": False, "reason": "unknown_action_contract"}
        if report["status"] == "INEQUIVALENT":
            fresh = equivalent(source, target, allow_transpose=action["allow_transpose"],
                               allow_frobenius=action["allow_frobenius"],
                               max_candidates=max_candidates, max_seconds=max_seconds)
            if fresh["status"] == "INCONCLUSIVE":
                return {"verified": None, "reason": "replay_limit"}
            ok = fresh["status"] == "INEQUIVALENT" and fresh["reason"] == report["reason"]
            if ok:
                ok = (report["certificate"] == fresh["certificate"])
            return {"verified": ok, "reason": "negative_decision_replayed"}
        if report["status"] != "EQUIVALENT":
            return {"verified": False, "reason": "unknown_status"}
        if _shape(source) != _shape(target) or len(a) != len(b):
            return {"verified": False, "reason": "shape_or_dimension_mismatch"}
        cert = report["certificate"]
        m, n = _shape(source)
        t, exponent = cert["transpose"], cert["frobenius_power"]
        if type(t) is not bool or (t and (not action["allow_transpose"] or m != n)):
            return {"verified": False, "reason": "disallowed_transpose"}
        if type(exponent) is not int or not 0 <= exponent < (f.degree if action["allow_frobenius"] else 1):
            return {"verified": False, "reason": "disallowed_automorphism"}
        permutation, scalars = cert["permutation"], cert["scalars"]
        if (not isinstance(permutation, list) or any(type(i) is not int for i in permutation)
                or sorted(permutation) != list(range(len(a))) or len(scalars) != len(a)):
            return {"verified": False, "reason": "invalid_matching"}
        for matrix, size in ((cert["P"], m), (cert["Q"], n)):
            if len(matrix) != size or any(len(row) != size for row in matrix):
                return {"verified": False, "reason": "wrong_transporter_size"}
            for row in matrix:
                for x in row:
                    f.element(x)
        for c in scalars:
            f.element(c)
            if not c:
                return {"verified": False, "reason": "zero_generator_scalar"}
        # Library Gaussian ranks/multiplication are independent of the search helpers.
        import numpy as np
        p, q = f.GF(cert["P"]), f.GF(cert["Q"])
        if np.linalg.matrix_rank(p) != m or np.linalg.matrix_rank(q) != n:
            return {"verified": False, "reason": "singular_transporter"}
        for i, j in enumerate(permutation):
            matrix = f.GF(a[i]) ** (f.p**exponent)
            if t:
                matrix = matrix.T
            if not np.array_equal(p @ matrix @ q, f.GF(scalars[i]) * f.GF(b[j])):
                return {"verified": False, "reason": "false_matrix_identity"}
        return {"verified": True, "reason": "every_transporter_identity_replayed"}
    except (ValueError, TypeError, KeyError, IndexError, StopIteration, ZeroDivisionError, AttributeError):
        return {"verified": False, "reason": "malformed_or_false_certificate"}


def _ordered_right_canonical(points, f):
    """Canonical ordered projective configuration, using a support forest."""
    s = len(points[0])
    basis = _basis_indices(points, f)
    inverse = _inverse(_columns([points[i] for i in basis]), f)
    coordinates = [_matvec(inverse, point, f) for point in points]
    parent = list(range(s))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    graph = [[] for _ in range(s)]
    for col in coordinates:
        support = [i for i, x in enumerate(col) if x]
        first = support[0]
        for j in support[1:]:
            if find(first) != find(j):
                parent[find(j)] = find(first)
                ratio = f.mul(col[first], f.inv(col[j]))
                graph[first].append((j, ratio))
                graph[j].append((first, f.inv(ratio)))
    beta = [None] * s
    for root in range(s):
        if beta[root] is not None:
            continue
        beta[root], stack = 1, [root]
        while stack:
            i = stack.pop()
            for j, ratio in graph[i]:
                if beta[j] is None:
                    beta[j] = f.mul(beta[i], ratio)
                    stack.append(j)
    diagonal = [[beta[i] if i == j else 0 for j in range(s)] for i in range(s)]
    transform = _matmul(diagonal, inverse, f)
    canonical = [list(_normalize(_matvec(transform, x, f), f)) for x in points]
    return canonical, transform


def canonical_form(problem, *, allow_transpose=False, allow_frobenius=False,
                   max_candidates=100000, max_seconds=None):
    """Complete canonical representative, or no key when work is interrupted."""
    budget = _Budget(max_candidates, max_seconds)
    action = _action(allow_transpose, allow_frobenius)
    f, matrices, canonical = prepare(problem)
    m, n = _shape(canonical)
    out = {"schema_version": 1, "method": "paired_projective_canonical",
           "problem": canonical, "action": action}

    def finish(status, reason, **extra):
        return {**out, "status": status, "reason": reason, "stats": budget.stats(), **extra}

    try:
        out["input_check"] = _validate(canonical, budget)
        if not matrices:
            key = {"field": f.metadata(), "shape": [m, n], "action": action, "pairs": []}
            return finish("COMPLETE", "zero_space", canonical_key=json.dumps(key, sort_keys=True),
                          representative=canonical,
                          transporter={"P": _identity(m), "Q": _identity(n), "permutation": [],
                                       "scalars": [], "transpose": False, "frobenius_power": 0})
        best = best_record = None
        for variant, transposed, exponent in _variants(matrices, f, allow_transpose,
                                                      allow_frobenius, m == n):
            u, v, eu, fv, r, s = _factor_data(variant, m, n, f)
            # Many left projectivities induce the same ordering. The ordered
            # right object depends only on that ordering, so reuse its complete
            # normalization without omitting any left candidate.
            right_cache = {}
            for indices in permutations(range(len(u)), r):
                basis = [u[i] for i in indices]
                if rank(basis, f) != r:
                    continue
                inverse = _inverse(_columns(basis), f)
                for tail in product(range(1, f.q), repeat=r-1):
                    budget.tick()
                    scales = (1,) + tail
                    left_map = [[f.mul(scales[i], z) for z in row]
                                for i, row in enumerate(inverse)]
                    left_points = [_normalize(_matvec(left_map, x, f), f) for x in u]
                    order = sorted(range(len(u)), key=lambda i: left_points[i])
                    ordered_left = [list(left_points[i]) for i in order]
                    order_key = tuple(order)
                    normalized = right_cache.get(order_key)
                    if normalized is None:
                        normalized = _ordered_right_canonical([v[i] for i in order], f)
                        if len(right_cache) < 8192:
                            right_cache[order_key] = normalized
                    ordered_right, right_map = normalized
                    serial = (r, s, tuple(tuple(row) for row in ordered_left),
                              tuple(tuple(row) for row in ordered_right))
                    if best is None or serial < best:
                        best = serial
                        p = _padded_map(left_map, m, eu, f)
                        q = _transpose(_padded_map(right_map, n, fv, f))
                        generators = [
                            [[f.mul(x, y) for y in right] + [0]*(n-s) for x in left]
                            + [[0]*n for _ in range(m-r)]
                            for left, right in zip(ordered_left, ordered_right)]
                        permutation = [order.index(i) for i in range(len(order))]
                        cert = _certificate(variant, generators, p, q, permutation,
                                            transposed, exponent, f)
                        best_record = (generators, cert)
        generators, cert = best_record
        key = {"field": f.metadata(), "shape": [m, n], "action": action,
               "essential_shape": list(best[:2]), "left": best[2], "right": best[3]}
        return finish("COMPLETE", "all_basis_images_exhausted",
                      canonical_key=json.dumps(key, sort_keys=True, separators=(",", ":")),
                      representative={"field": f.metadata(), "generators": generators},
                      transporter=cert)
    except _Limit:
        return finish("INCONCLUSIVE", "work_or_time_limit")
    except _Invalid as exc:
        return finish("INVALID_INPUT", "input_is_not_a_good_basis", input_reason=str(exc))


def classify_families(problems, *, allow_transpose=False, allow_frobenius=False,
                      max_candidates=100000, max_seconds=None, domain_complete=False):
    """Deduplicate supplied families; completeness never implies all ambient sets.

    domain_complete=True means the caller supplied its entire declared finite
    domain. The returned scope is always explicitly the supplied family list.
    """
    if not isinstance(problems, list):
        raise ValueError("problems must be a list")
    if type(domain_complete) is not bool:
        raise ValueError("domain_complete must be boolean")
    budget = _Budget(max_candidates, max_seconds)
    action = _action(allow_transpose, allow_frobenius)
    classes, by_key, records = [], {}, []
    for index, problem in enumerate(problems):
        remaining = max(0, budget.maximum-budget.count)
        out = canonical_form(problem, allow_transpose=allow_transpose,
                             allow_frobenius=allow_frobenius,
                             max_candidates=remaining, max_seconds=budget.remaining())
        budget.count += out["stats"]["candidates_checked"]
        record = {"input_index": index, "status": out["status"], "reason": out["reason"]}
        if out["status"] == "COMPLETE":
            key = out["canonical_key"]
            if key not in by_key:
                by_key[key] = len(classes)
                classes.append({"class_index": len(classes), "canonical_key": key,
                                "representative": out["representative"], "input_indices": []})
            group = by_key[key]
            classes[group]["input_indices"].append(index)
            record.update(class_index=group, transporter=out["transporter"])
        records.append(record)
        if out["status"] == "INCONCLUSIVE":
            break
    processed = len(records)
    complete = processed == len(problems) and all(r["status"] != "INCONCLUSIVE" for r in records)
    return {"schema_version": 1, "method": "completed_canonical_keys", "action": action,
            "problems": deepcopy(problems),
            "status": "COMPLETE_CLASSIFICATION" if complete and domain_complete else "PARTIAL_CLASSIFICATION",
            "scope": "supplied_family_list_only", "declared_domain_complete": domain_complete,
            "all_supplied_inputs_processed": complete, "families_supplied": len(problems),
            "families_processed": processed, "class_count": len(classes), "classes": classes,
            "records": records, "stats": budget.stats()}


def verify_canonical(report, *, max_candidates=100000, max_seconds=None):
    """Replay the map and completed minimum under the declared action."""
    try:
        if report["status"] == "INCONCLUSIVE":
            return {"verified": None, "reason": "no_completed_canonical_key"}
        if (report["status"] != "COMPLETE" or type(report.get("schema_version")) is not int
                or report["schema_version"] != 1):
            return {"verified": False, "reason": "unknown_canonical_claim"}
        transported = verify_equivalence({
            "schema_version": 1, "status": "EQUIVALENT", "source": report["problem"],
            "target": report["representative"], "action": report["action"],
            "certificate": report["transporter"],
        })
        if transported["verified"] is not True:
            return transported
        fresh = canonical_form(report["problem"],
                               allow_transpose=report["action"]["allow_transpose"],
                               allow_frobenius=report["action"]["allow_frobenius"],
                               max_candidates=max_candidates, max_seconds=max_seconds)
        if fresh["status"] == "INCONCLUSIVE":
            return {"verified": None, "reason": "replay_limit"}
        ok = (fresh["status"] == "COMPLETE" and
              fresh["canonical_key"] == report["canonical_key"] and
              fresh["representative"] == report["representative"])
        return {"verified": ok, "reason": "completed_canonical_minimum_replayed"}
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        return {"verified": False, "reason": "malformed_or_false_canonical_report"}


def small_class_representatives(q, m, n, k, *, allow_transpose=False, allow_frobenius=False):
    """Complete good-space classes for k<=3 in every ambient shape.

    Counts follow the independent-factor theorem for k2 and the four essential
    rank pairs for k3. These are mathematical classifications, not sampled sets.
    """
    from .field import Field
    f = Field(q)
    action = _action(allow_transpose, allow_frobenius)
    if any(type(x) is not int or x < 1 for x in (m, n)) or type(k) is not int or k < 0:
        raise ValueError("positive matrix dimensions and a nonnegative set size are required")
    out = {"q": q, "rows": m, "columns": n, "dimension": k, "action": action,
           "scope": "all_good_spaces_at_declared_field_shape_and_dimension",
           "proof": "k<=3 paired-factor rank classification; EQUIVALENCE_RESEARCH.md section7"}

    def outer(u, v):
        u, v = u + [0]*(m-len(u)), v + [0]*(n-len(v))
        return [[f.mul(x, y) for y in v] for x in u]

    representatives, labels = [], []
    if k == 0:
        representatives.append({"field": f.metadata(), "rows": m, "columns": n, "generators": []})
        labels.append("zero_space")
    elif k == 1:
        representatives.append({"field": f.metadata(), "generators": [outer([1], [1])]})
        labels.append("one_rank_one_direction")
    elif k == 2 and m >= 2 and n >= 2:
        representatives.append({"field": f.metadata(), "generators":
                                [outer([1, 0], [1, 0]), outer([0, 1], [0, 1])]})
        labels.append("essential_ranks_2_2")
    elif k == 3:
        two = [[1, 0], [0, 1], [1, 1]]
        three = _identity(3)
        candidates = [(2, 2), (2, 3), (3, 2), (3, 3)]
        for r, s in candidates:
            if r > m or s > n or (r == s == 2 and q != 2):
                continue
            if allow_transpose and m == n and (r, s) == (3, 2):
                continue
            left, right = two if r == 2 else three, two if s == 2 else three
            representatives.append({"field": f.metadata(), "generators":
                                    [outer(u, v) for u, v in zip(left, right)]})
            labels.append(f"essential_ranks_{r}_{s}")
    elif k > 3:
        return {**out, "status": "UNAVAILABLE", "class_count": None,
                "representatives": [], "reason": "small_generator_theorem_covers_k_at_most_3"}
    return {**out, "status": "COMPLETE_CLASSIFICATION", "class_count": len(representatives),
            "representatives": representatives, "labels": labels}
