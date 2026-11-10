"""Cross-check ambient membership against coefficient enumeration."""

from itertools import product
import random

import pytest

from rankone.binary_membership import binary_membership_check
from rankone.field import Field
from rankone.reference import library_combine, library_rank, reference_check


def problem(matrices, q=2):
    return {"field": {"order": q}, "generators": matrices}


def outer(u, v):
    return [[x * y for y in v] for x in u]


E11 = [[1, 0], [0, 0]]
E22 = [[0, 0], [0, 1]]
J = [[1, 1], [1, 1]]


def test_binary_triple_checks_complete_ambient_set():
    out = binary_membership_check(problem([E11, E22, J]))
    assert out["status"] == "VALID" and out["dimension"] == 3
    assert out["candidates_examined"] == out["candidates_total"] == 9
    assert {tuple(sum(a, [])) for a in out["rank_one_members"]} == {
        tuple(sum(a, [])) for a in (E11, E22, J)
    }
    assert out["counterexample"] is None


def test_all_49_singleton_three_by_three_inputs():
    vectors = [v for v in product((0, 1), repeat=3) if any(v)]
    for u in vectors:
        for v in vectors:
            matrix = outer(u, v)
            out = binary_membership_check(problem([matrix]))
            assert out["status"] == "VALID"
            assert out["candidates_examined"] == out["candidates_total"] == 49
            assert out["rank_one_members"] == [matrix]


def test_dependent_seven_symmetric_points_rejected():
    vectors = [v for v in product((0, 1), repeat=3) if any(v)]
    matrices = [outer(u, u) for u in vectors]
    out = binary_membership_check(problem(matrices))
    assert out["status"] == "INVALID" and out["reason"] == "dependent_generators"
    assert out["dimension"] == 6
    relation = out["counterexample"]["coefficients"]
    assert any(relation)
    assert library_rank(library_combine(matrices, relation, Field(2)), Field(2)) == 0


@pytest.mark.parametrize("matrix,rank", [([[0, 0], [0, 0]], 0),
                                         ([[1, 0], [0, 1]], 2)])
def test_wrong_generator_rank(matrix, rank):
    out = binary_membership_check(problem([matrix]))
    assert out["status"] == "INVALID" and out["reason"] == "generator_not_rank_one"
    assert out["counterexample"] == {"index": 0, "rank": rank}


def test_extra_rank_one_has_replayable_coefficients_and_factors():
    matrices = [E11, [[0, 1], [0, 0]]]
    out = binary_membership_check(problem(matrices))
    assert out["status"] == "INVALID" and out["reason"] == "extra_rank_one"
    witness = out["counterexample"]
    assert witness["coefficients"] == [1, 1]
    assert library_combine(matrices, witness["coefficients"], Field(2)) == witness["matrix"]
    assert outer(witness["u"], witness["v"]) == witness["matrix"]
    assert witness["matrix"] not in matrices


@pytest.mark.parametrize("budget", [0, 1, 8])
def test_budget_never_accepts_an_unfinished_valid_instance(budget):
    out = binary_membership_check(problem([E11, E22, J]), max_candidates=budget)
    assert out["status"] == "INCONCLUSIVE"
    assert out["candidates_examined"] == budget < out["candidates_total"]
    assert binary_membership_check(problem([E11, E22, J]), max_candidates=9)["status"] == "VALID"


@pytest.mark.parametrize("q", [3, 4, 5])
def test_nonbinary_fields_rejected(q):
    with pytest.raises(ValueError, match="field order 2"):
        binary_membership_check(problem([E11], q))


@pytest.mark.parametrize("budget", [-1, True, 1.5])
def test_invalid_budgets_rejected(budget):
    with pytest.raises(ValueError):
        binary_membership_check(problem([E11]), max_candidates=budget)


def test_seeded_rectangular_families_against_full_coefficient_reference():
    rng = random.Random(202610223)
    field = Field(2)
    for _ in range(160):
        m, n, k = rng.randrange(1, 5), rng.randrange(1, 5), rng.randrange(1, 8)
        matrices = [outer([rng.randrange(2) for _ in range(m)],
                          [rng.randrange(2) for _ in range(n)]) for _ in range(k)]
        out = binary_membership_check(problem(matrices))
        reference = reference_check(problem(matrices))
        assert out["status"] == reference["status"]
        if out["reason"] == "extra_rank_one":
            witness = out["counterexample"]
            assert library_combine(matrices, witness["coefficients"], field) == witness["matrix"]
            assert library_rank(witness["matrix"], field) == 1
            assert sum(witness["coefficients"]) >= 2


def test_twenty_dimensional_five_by_five_witness_by_all_ambient_points():
    # Fixed witness from the October22 search, independently expanded here.
    # This test does not call the search or enumerate its 2**20 span elements.
    pairs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 8), (7, 10),
             (8, 6), (9, 9), (10, 11), (11, 7), (12, 16), (13, 17),
             (14, 22), (16, 12), (17, 13), (18, 18), (21, 27),
             (25, 19), (28, 21)]
    matrices = [outer([(u >> i) & 1 for i in range(5)],
                      [(v >> j) & 1 for j in range(5)]) for u, v in pairs]
    out = binary_membership_check(problem(matrices))
    assert out["status"] == "VALID"
    assert out["dimension"] == out["generator_count"] == 20
    assert out["candidates_examined"] == out["candidates_total"] == 961
    assert {tuple(sum(a, [])) for a in out["rank_one_members"]} == {
        tuple(sum(a, [])) for a in matrices
    }
