"""Deliberately simple independent oracle: full coefficients, library rank."""

from itertools import product

from .problem import prepare, result


def library_rank(matrix, f):
    if len(matrix)==0:
        return 0
    reduced = f.GF(matrix).row_reduce()
    return sum(any(int(x) for x in row) for row in reduced)


def library_combine(generators, coefficients, f):
    a = f.GF(generators)
    weights = f.GF(coefficients)
    return (weights[:, None, None] * a).sum(axis=0).tolist()


def reference_check(problem, max_vectors=100000):
    if type(max_vectors) is not int or max_vectors < 0:
        raise ValueError("max_vectors must be a nonnegative integer")
    f, matrices, canonical = prepare(problem)
    stats = {"vectors_checked": 0}
    def finish(status, reason, certificate):
        return result(canonical, "reference-full", status, reason, certificate, stats)
    for i, matrix in enumerate(matrices):
        r = library_rank(matrix, f)
        if r != 1:
            return finish("INVALID", "generator_not_rank_one", {"index": i, "rank": r})
    rows = [sum(matrix, []) for matrix in matrices]
    if library_rank(rows, f) != len(matrices):
        relation = f.GF(rows).left_null_space()[0].tolist()
        return finish("INVALID", "dependent_generators", {"coefficients": relation})
    k = len(matrices)
    for coefficients in product(range(f.q), repeat=k):
        if sum(x != 0 for x in coefficients) < 2:
            continue
        if stats["vectors_checked"] >= max_vectors:
            return finish("INCONCLUSIVE", "work_limit", {})
        matrix = library_combine(matrices, coefficients, f)
        stats["vectors_checked"] += 1
        if library_rank(matrix, f) == 1:
            return finish("INVALID", "extra_rank_one", {
                "coefficients": list(coefficients), "matrix": matrix})
    return finish("VALID", "exhausted_full_coefficients", {
        "mixed_vectors": f.q ** k - 1 - k * (f.q - 1)})
