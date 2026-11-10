"""Exact rank-one census by annihilator fibres and separate certificate replay.

The producer uses the field library's null spaces. Replay uses integer-label
field operations and Gaussian ranks, and never calls the producer. Both share
the declared field model, which is not a formal proof kernel.
"""

from itertools import product
from time import monotonic

from .linalg import combine, elimination, factors, flatten, rank, rank_one
from .problem import prepare, result


def projective_vectors(field, dimension):
    """Every nonzero scalar direction exactly once, first nonzero entry one."""
    for first in range(dimension):
        for tail in product(range(field.q), repeat=dimension - first - 1):
            yield [0] * first + [1] + list(tail)


def normalize(vector, field):
    pivot = next((x for x in vector if x), None)
    if pivot is None:
        raise ValueError("zero has no projective normalization")
    inverse = field.inv(pivot)
    return tuple(field.mul(x, inverse) for x in vector)


def _shape(matrices, canonical):
    return ((len(matrices[0]), len(matrices[0][0])) if matrices else
            (canonical["rows"], canonical["columns"]))


def _transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def _library_rank(array):
    return len(array.row_space())


def _limit(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def fibre_check(problem, *, max_fibres=100000, max_seconds=None):
    """Decide the exact independent-basis/no-extra condition with certificates.

    A successful check visits N_min(m,n) fibres. Limits always yield
    INCONCLUSIVE. The empty basis is accepted with explicit positive shape.
    """
    _limit(max_fibres, "max_fibres")
    if max_seconds is not None and (type(max_seconds) not in (int, float)
                                    or not 0 <= max_seconds < float("inf")):
        raise ValueError("max_seconds must be finite and nonnegative")
    started = monotonic()
    f, matrices, canonical = prepare(problem)
    m, n = _shape(matrices, canonical)
    k = len(matrices)
    transposed = m > n
    rows, columns = (n, m) if transposed else (m, n)
    total = (f.q ** rows - 1) // (f.q - 1)
    stats = {"fibres_checked": 0, "fibres_total": total,
             "rank_one_directions": 0, "dimension": k, "rows": m,
             "columns": n, "transposed": transposed,
             "ambient_rank_one_directions":
                 ((f.q ** m - 1) // (f.q - 1)) * ((f.q ** n - 1) // (f.q - 1))}

    def finish(status, reason, certificate):
        return result(canonical, "annihilator-fibre", status, reason, certificate,
                      {**stats, "elapsed_seconds": monotonic() - started,
                       "limits": {"max_fibres": max_fibres, "max_seconds": max_seconds}})

    def expired():
        return max_seconds is not None and monotonic() - started >= max_seconds

    for index, matrix in enumerate(matrices):
        if expired():
            return finish("INCONCLUSIVE", "time_limit", {})
        actual_rank = _library_rank(f.GF(matrix))
        if actual_rank != 1:
            return finish("INVALID", "generator_not_rank_one",
                          {"index": index, "rank": actual_rank})
    flat = f.GF([flatten(a) for a in matrices]) if k else f.GF.Zeros((0, m*n))
    if _library_rank(flat) != k:
        relation = flat.T.null_space()[0].tolist()
        return finish("INVALID", "dependent_generators", {"coefficients": relation})
    if expired():
        return finish("INCONCLUSIVE", "time_limit", {})

    pairs = [factors(a, f) for a in matrices]
    seen_left, seen_right = {}, {}
    for index, (u, v) in enumerate(pairs):
        for seen, vector in ((seen_left, u), (seen_right, v)):
            key = normalize(vector, f)
            if key in seen:
                coefficients = [0] * k
                coefficients[seen[key]] = coefficients[index] = 1
                matrix = combine(matrices, coefficients, f)
                cu, cv = factors(matrix, f)
                return finish("INVALID", "extra_rank_one", {
                    "coefficients": coefficients, "matrix": matrix, "u": cu, "v": cv,
                    "source": "shared_factor_pair"})
            seen[key] = index

    oriented = [_transpose(a) for a in matrices] if transposed else matrices
    oriented_flat = f.GF([flatten(a) for a in oriented]) if k else f.GF.Zeros((0, rows*columns))
    if expired():
        return finish("INCONCLUSIVE", "time_limit", {})
    checks = oriented_flat.null_space().reshape(m*n-k, rows, columns)
    cert = {"transposed": transposed, "shape": [rows, columns],
            "parity_checks": checks.tolist(), "fibres": [], "rank_one_directions": 0}
    allowed = {}
    for matrix in oriented:
        u, v = factors(matrix, f)
        allowed[normalize(u, f)] = normalize(v, f)
    for u in projective_vectors(f, rows):
        if expired():
            return finish("INCONCLUSIVE", "time_limit", cert)
        if stats["fibres_checked"] >= max_fibres:
            return finish("INCONCLUSIVE", "fibre_limit", cert)
        equations = (f.GF([f.GF(u) @ b for b in checks]) if len(checks)
                     else f.GF.Zeros((0, columns)))
        kernel = equations.null_space()
        d = len(kernel)
        stats["fibres_checked"] += 1
        stats["rank_one_directions"] += (f.q**d - 1) // (f.q - 1)
        cert["fibres"].append({"left": u, "nullity": d, "kernel_basis": kernel.tolist()})
        cert["rank_one_directions"] = stats["rank_one_directions"]
        target = allowed.get(tuple(u))
        extra = next((v for v in kernel.tolist()
                      if target is None or normalize(v, f) != target), None)
        if extra is not None:
            array = f.GF(u)[:, None] * f.GF(extra)[None, :]
            matrix = array.T.tolist() if transposed else array.tolist()
            # G^T has full column rank. Its reduced augmented system has k pivots.
            rhs = f.GF(flatten(matrix))[:, None]
            augmented = f.GF.Zeros((m*n, k+1))
            augmented[:, :k] = flat.T
            augmented[:, k:] = rhs
            reduced = augmented.row_reduce()
            coefficients = reduced[:k, k].tolist()
            # Use the same coefficient-direction convention as the older
            # checker, making the first active coefficient one for display.
            scale=f.inv(next(x for x in coefficients if x))
            coefficients=[f.mul(x,scale) for x in coefficients]
            matrix=combine(matrices,coefficients,f)
            cu, cv = factors(matrix, f)
            return finish("INVALID", "extra_rank_one", {
                "coefficients": coefficients, "matrix": matrix, "u": cu, "v": cv,
                "source": "annihilator_fibre"})
    if stats["rank_one_directions"] != k:
        raise ArithmeticError("fibre census disagreed with rank-one basis coverage")
    return finish("VALID", "all_rank_one_fibres_certified", cert)


def _dot(a, b, f):
    value = 0
    for x, y in zip(a, b):
        value = f.add(value, f.mul(x, y))
    return value


def _matrix(value, rows, columns, f):
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError("wrong matrix row count")
    for row in value:
        if not isinstance(row, list) or len(row) != columns:
            raise ValueError("wrong matrix column count")
        for x in row:
            f.element(x)
    return value


def verify_fibre(report, *, max_fibres=100000):
    """Replay complete evidence without calling fibre_check or library kernels."""
    _limit(max_fibres, "max_fibres")
    try:
        return _verify_fibre(report, max_fibres)
    except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError, AttributeError):
        return {"verified": False, "reason": "malformed_or_false_certificate"}


def _verify_fibre(report, max_fibres):
    def answer(ok, reason):
        return {"verified": ok, "reason": reason, "claim": report.get("status")}

    if type(report["schema_version"]) is not int or report["schema_version"] != 1:
        return answer(False, "unknown_schema")
    if report["method"] != "annihilator-fibre":
        return answer(False, "unknown_method")
    f, matrices, canonical = prepare(report["problem"])
    m, n = _shape(matrices, canonical)
    k = len(matrices)
    cert = report["certificate"]
    if report["status"] == "INCONCLUSIVE":
        return answer(None, "no_mathematical_decision_to_verify")
    if report["status"] not in ("VALID", "INVALID"):
        return answer(False, "unknown_status")
    if report["status"] == "INVALID":
        if report["reason"] == "generator_not_rank_one":
            index = cert["index"]
            if type(index) is not int or not 0 <= index < k:
                return answer(False, "invalid_index")
            r = rank(matrices[index], f)
            return answer(type(cert["rank"]) is int and r == cert["rank"] and r != 1,
                          "generator_rank_replayed")
        coefficients = cert["coefficients"]
        if not isinstance(coefficients, list) or len(coefficients) != k:
            return answer(False, "invalid_coefficients")
        for x in coefficients:
            f.element(x)
        if not any(coefficients):
            return answer(False, "zero_coefficient_witness")
        matrix = combine(matrices, coefficients, f)
        if report["reason"] == "dependent_generators":
            return answer(not any(flatten(matrix)), "dependence_replayed")
        if report["reason"] != "extra_rank_one":
            return answer(False, "unknown_invalid_reason")
        if any(not rank_one(a, f) for a in matrices):
            return answer(False, "invalid_generator")
        if rank([flatten(a) for a in matrices], f) != k:
            return answer(False, "dependent_basis")
        if sum(x != 0 for x in coefficients) < 2:
            return answer(False, "allowed_singleton_direction")
        if matrix != cert["matrix"] or not rank_one(matrix, f):
            return answer(False, "false_extra_direction")
        u, v = cert["u"], cert["v"]
        if not isinstance(u, list) or not isinstance(v, list) or len(u) != m or len(v) != n:
            return answer(False, "wrong_factor_sizes")
        for x in u + v:
            f.element(x)
        if [[f.mul(x, y) for y in v] for x in u] != matrix:
            return answer(False, "false_factorization")
        key = normalize(flatten(matrix), f)
        return answer(all(normalize(flatten(a), f) != key for a in matrices),
                      "extra_rank_one_replayed")

    if report["reason"] != "all_rank_one_fibres_certified":
        return answer(False, "unknown_valid_reason")
    if any(not rank_one(a, f) for a in matrices):
        return answer(False, "invalid_generator")
    if rank([flatten(a) for a in matrices], f) != k:
        return answer(False, "dependent_basis")
    transposed = m > n
    rows, columns = (n, m) if transposed else (m, n)
    if type(cert["transposed"]) is not bool or cert["transposed"] != transposed:
        return answer(False, "incorrect_orientation")
    if cert["shape"] != [rows, columns]:
        return answer(False, "incorrect_shape")
    total = (f.q**rows - 1) // (f.q - 1)
    if total > max_fibres:
        return answer(None, "replay_limit")
    oriented = [_transpose(a) for a in matrices] if transposed else matrices
    flat = [flatten(a) for a in oriented]
    checks = cert["parity_checks"]
    if not isinstance(checks, list) or len(checks) != m*n-k:
        return answer(False, "incomplete_annihilator")
    for b in checks:
        _matrix(b, rows, columns, f)
    check_rows = [flatten(b) for b in checks]
    if rank(check_rows, f) != m*n-k:
        return answer(False, "dependent_annihilator")
    if any(_dot(a, b, f) for a in flat for b in check_rows):
        return answer(False, "false_annihilator")
    records = cert["fibres"]
    if not isinstance(records, list) or len(records) != total:
        return answer(False, "incomplete_fibre_coverage")
    count = 0
    for expected, record in zip(projective_vectors(f, rows), records):
        if record["left"] != expected:
            return answer(False, "missing_duplicate_or_reordered_fibre")
        u = record["left"]
        for x in u:
            f.element(x)
        equations = [[_dot(u, [b[i][j] for i in range(rows)], f)
                      for j in range(columns)] for b in checks]
        kernel = record["kernel_basis"]
        if not isinstance(kernel, list):
            return answer(False, "invalid_kernel")
        _matrix(kernel, len(kernel), columns, f)
        d = len(kernel)
        if type(record["nullity"]) is not int or record["nullity"] != d:
            return answer(False, "incorrect_nullity")
        if rank(kernel, f) != d or rank(equations, f) + d != columns:
            return answer(False, "incomplete_or_dependent_kernel")
        if any(_dot(eq, v, f) for eq in equations for v in kernel):
            return answer(False, "false_kernel")
        count += (f.q**d - 1) // (f.q - 1)
    if type(cert["rank_one_directions"]) is not int or cert["rank_one_directions"] != count:
        return answer(False, "incorrect_direction_count")
    if count != k:
        return answer(False, "extra_rank_one_directions")
    stats = report["stats"]
    expected_stats = {"fibres_checked": total, "fibres_total": total,
                      "rank_one_directions": count, "dimension": k, "rows": m, "columns": n,
                      "ambient_rank_one_directions":
                          ((f.q**m-1)//(f.q-1))*((f.q**n-1)//(f.q-1))}
    if any(type(stats[key]) is not int or stats[key] != value
           for key, value in expected_stats.items()):
        return answer(False, "inconsistent_stats")
    if type(stats["transposed"]) is not bool or stats["transposed"] != transposed:
        return answer(False, "inconsistent_orientation_stats")
    return answer(True, "all_annihilator_fibres_replayed")
