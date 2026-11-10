"""Exact elimination and a pivot-based rank-one test, with no floating point."""


def elimination(rows, field, track=False):
    a = [list(row) for row in rows]
    if not a:
        return 0, None
    count, width = len(a), len(a[0])
    t = [[int(i == j) for j in range(count)] for i in range(count)] if track else None
    pivot = 0
    for col in range(width):
        chosen = next((i for i in range(pivot, count) if a[i][col]), None)
        if chosen is None:
            continue
        a[pivot], a[chosen] = a[chosen], a[pivot]
        if track:
            t[pivot], t[chosen] = t[chosen], t[pivot]
        scale = field.inv(a[pivot][col])
        a[pivot] = [field.mul(x, scale) for x in a[pivot]]
        if track:
            t[pivot] = [field.mul(x, scale) for x in t[pivot]]
        for i in range(count):
            if i == pivot or not a[i][col]:
                continue
            scale = a[i][col]
            a[i] = [field.sub(x, field.mul(scale, y)) for x, y in zip(a[i], a[pivot])]
            if track:
                t[i] = [field.sub(x, field.mul(scale, y)) for x, y in zip(t[i], t[pivot])]
        pivot += 1
        if pivot == count:
            break
    relation = next((t[i] for i in range(count) if not any(a[i])), None) if track else None
    return pivot, relation


def rank(matrix, field):
    return elimination(matrix, field)[0]


def rank_one(matrix, field):
    """Nonzero pivot plus all pivot minors zero is equivalent to rank exactly 1."""
    pivot = next(((i, j) for i, row in enumerate(matrix)
                  for j, x in enumerate(row) if x), None)
    if pivot is None:
        return False
    r, c = pivot
    p = matrix[r][c]
    return all(field.mul(p, x) == field.mul(matrix[i][c], matrix[r][j])
               for i, row in enumerate(matrix) for j, x in enumerate(row))


def factors(matrix, field):
    r, c = next((i, j) for i, row in enumerate(matrix)
                for j, x in enumerate(row) if x)
    u = [row[c] for row in matrix]
    v = [field.mul(x, field.inv(matrix[r][c])) for x in matrix[r]]
    return u, v


def flatten(matrix):
    return [x for row in matrix for x in row]


def combine(generators, coefficients, field):
    m, n = len(generators[0]), len(generators[0][0])
    result = [[0] * n for _ in range(m)]
    for coefficient, matrix in zip(coefficients, generators):
        if coefficient:
            for i in range(m):
                for j in range(n):
                    result[i][j] = field.add(result[i][j], field.mul(coefficient, matrix[i][j]))
    return result
