from itertools import product

import pytest

from rankone.field import Field
from rankone.linalg import elimination, factors, rank, rank_one


@pytest.mark.parametrize("q", [2, 3, 4, 5, 8, 9])
def test_field_axioms_and_library_agreement(q):
    f = Field(q)
    for a, b in product(range(q), repeat=2):
        assert f.add(a, b) == int(f.GF(a) + f.GF(b))
        assert f.mul(a, b) == int(f.GF(a) * f.GF(b))
        assert f.sub(a, b) == int(f.GF(a) - f.GF(b))
        for c in range(q):
            assert f.mul(a, f.add(b, c)) == f.add(f.mul(a, b), f.mul(a, c))
    for a in range(1, q):
        assert f.mul(a, f.inv(a)) == 1


@pytest.mark.parametrize("q", [0, 1, 6, 10, 12, True, 2.0])
def test_invalid_orders(q):
    with pytest.raises((ValueError, TypeError)):
        Field(q)


def test_extension_is_not_mod_four():
    f = Field(4, [1, 1, 1])
    assert f.add(1, 1) == 0
    assert f.mul(2, 2) == 3  # alpha**2 = alpha + 1
    for invalid in [-1, 4, 1.0, True, "1"]:
        with pytest.raises(ValueError):
            f.element(invalid)
    with pytest.raises(ValueError):
        Field(4, [1, 0, 1])  # (x+1)^2 is reducible in characteristic 2


@pytest.mark.parametrize("q", [2, 3, 4])
def test_all_two_by_two_ranks_against_library(q):
    f = Field(q)
    for entries in product(range(q), repeat=4):
        matrix = [list(entries[:2]), list(entries[2:])]
        reduced = f.GF(matrix).row_reduce()
        expected = sum(any(int(x) for x in row) for row in reduced)
        assert rank(matrix, f) == expected
        assert rank_one(matrix, f) == (expected == 1)
        if expected == 1:
            u, v = factors(matrix, f)
            assert [[f.mul(x, y) for y in v] for x in u] == matrix


def test_dependency_witness():
    f = Field(3)
    rows = [[1, 0, 1], [0, 1, 1], [1, 1, 2]]
    dimension, relation = elimination(rows, f, track=True)
    assert dimension == 2 and any(relation)
    assert all(sum(c * row[j] for c, row in zip(relation, rows)) % 3 == 0
               for j in range(3))
