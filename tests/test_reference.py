import pytest

from rankone.problem import prepare
from rankone.reference import library_combine, library_rank, reference_check


def problem(q, matrices):
    return {"field": {"order": q}, "generators": matrices}


E11 = [[1, 0], [0, 0]]
E22 = [[0, 0], [0, 1]]
J = [[1, 1], [1, 1]]


@pytest.mark.parametrize("q", [2, 3, 4, 5, 8, 9])
def test_reference_diagonal_and_shared_factor(q):
    assert reference_check(problem(q, [E11, E22]))["status"] == "VALID"
    bad = reference_check(problem(q, [E11, [[0, 1], [0, 0]]]))
    assert bad["reason"] == "extra_rank_one"
    f, matrices, _ = prepare(bad["problem"])
    witness = library_combine(matrices, bad["certificate"]["coefficients"], f)
    assert witness == bad["certificate"]["matrix"]
    assert library_rank(witness, f) == 1


def test_three_generator_characteristic_effect():
    assert reference_check(problem(2, [E11, E22, J]))["status"] == "VALID"
    bad = reference_check(problem(3, [E11, E22, J]))
    assert bad["status"] == "INVALID"
    assert all(bad["certificate"]["coefficients"])
    # All pairs pass: higher-support checking is indispensable.
    for pair in ([E11, E22], [E11, J], [E22, J]):
        assert reference_check(problem(3, pair))["status"] == "VALID"


def test_bad_generators_and_dependence():
    for matrix in ([[0, 0], [0, 0]], [[1, 0], [0, 1]]):
        assert reference_check(problem(2, [matrix]))["reason"] == "generator_not_rank_one"
    out = reference_check(problem(3, [E11, [[2, 0], [0, 0]]]))
    assert out["reason"] == "dependent_generators"
    f, matrices, _ = prepare(out["problem"])
    assert library_combine(matrices, out["certificate"]["coefficients"], f) == [[0, 0], [0, 0]]


def test_limit_is_not_success():
    assert reference_check(problem(2, [E11, E22]), max_vectors=0)["status"] == "INCONCLUSIVE"
    assert reference_check(problem(2, [E11]), max_vectors=0)["status"] == "VALID"


@pytest.mark.parametrize("matrices", [[], [[]], [[[1], [1, 0]]], [[[1]], [[1, 0]]], [[[True]]]])
def test_malformed_inputs(matrices):
    with pytest.raises(ValueError):
        reference_check(problem(2, matrices))
