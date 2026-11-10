from copy import deepcopy
from itertools import combinations, product
import random

import pytest

from rankone.checker import check, normalized_coefficients, rank_distribution, supports
from rankone.field import Field
from rankone.reference import reference_check
from test_reference import E11, E22, J, problem


@pytest.mark.parametrize("q,k", [(2, 4), (3, 3), (4, 3), (5, 3)])
def test_normalization_covers_each_scalar_orbit_once(q, k):
    f = Field(q)
    normalized = [tuple(c) for s in supports(k, 1) for c in normalized_coefficients(k, s, q)]
    assert len(set(normalized)) == len(normalized) == (q ** k - 1) // (q - 1)
    expanded = {tuple(f.mul(t, x) for x in c) for c in normalized for t in range(1, q)}
    assert expanded == {c for c in product(range(q), repeat=k) if any(c)}


@pytest.mark.parametrize("q", [2, 3])
def test_exhaustive_rank_one_two_by_two_families(q):
    f = Field(q)
    # Produce the oracle's projective rank-one candidates independently from factors.
    candidates = set()
    for u in product(range(q), repeat=2):
        for v in product(range(q), repeat=2):
            a = tuple(f.mul(x, y) for x in u for y in v)
            if any(a):
                inv = f.inv(next(x for x in a if x))
                candidates.add(tuple(f.mul(x, inv) for x in a))
    candidates = [[list(a[:2]), list(a[2:])] for a in sorted(candidates)]
    for size in (1, 2, 3):
        for family in combinations(candidates, size):
            instance = problem(q, list(family))
            reference = reference_check(instance)
            for method in ("hybrid", "normalized"):
                actual = check(instance, method=method)
                assert actual["status"] == reference["status"]
                assert actual["reason"] == reference["reason"] or actual["status"] == "VALID"


@pytest.mark.parametrize("q", [4, 5, 8, 9])
def test_seeded_rectangular_extension_cases(q):
    rng = random.Random(20261022 + q)
    f = Field(q)
    for _ in range(12):
        family = []
        for _ in range(rng.randrange(1, 5)):
            u = [rng.randrange(q) for _ in range(2)]
            v = [rng.randrange(q) for _ in range(3)]
            family.append([[f.mul(x, y) for y in v] for x in u])
        instance = problem(q, family)
        reference = reference_check(instance)
        assert reference["status"] != "INCONCLUSIVE"
        for method in ("hybrid", "normalized"):
            assert check(instance, method=method)["status"] == reference["status"]


def test_inconclusive_bound_falls_back_and_field_changes_answer():
    out = check(problem(2, [E11, E22, J]))
    assert out["status"] == "VALID"
    assert out["stats"]["directions_by_bound"] == 3
    assert out["stats"]["directions_checked"] == 1
    assert out["certificate"]["supports"][-1]["kind"] == "enumerated"
    out = check(problem(3, [E11, E22, J]))
    assert out["reason"] == "extra_rank_one"
    assert all(out["certificate"]["coefficients"])


def test_work_limits_and_exact_boundary():
    instance = problem(3, [E11, E22])
    assert check(instance, method="normalized", max_directions=0)["status"] == "INCONCLUSIVE"
    assert check(instance, max_supports=0)["reason"] == "support_limit"
    assert check(instance, max_seconds=0)["reason"] == "time_limit"
    assert check(instance, method="normalized", max_directions=2)["status"] == "VALID"
    assert check(instance, max_directions=0)["status"] == "VALID"  # proven, not sampled
    for bad in (-1, True, float("nan")):
        with pytest.raises(ValueError):
            check(instance, max_directions=bad)
    with pytest.raises(ValueError):
        check(instance, max_seconds=float("nan"))


def test_distribution_counts_matrices_not_directions():
    out = rank_distribution(problem(3, [E11, E22]))
    assert out["status"] == "COMPLETE"
    assert out["matrix_counts"] == {"0": 1, "1": 4, "2": 4}
    assert sum(out["matrix_counts"].values()) == 9
    assert rank_distribution(problem(3, [E11, E22]), max_directions=2)["status"] == "INCONCLUSIVE"
    with pytest.raises(ValueError):
        rank_distribution(problem(2, [E11, E11]))


def test_invariance_under_invertible_actions_permutation_and_rescaling():
    f = Field(4)
    original = [E11, E22, J]
    # Different nonzero scalar on each basis vector, and fixed invertible L,R.
    L, R = f.GF([[1, 2], [0, 1]]), f.GF([[1, 0], [3, 1]])
    changed = [(L @ (f.GF(a) * f.GF(i + 1)) @ R).tolist()
               for i, a in enumerate(original)]
    changed.reverse()
    baseline = check(problem(4, original))["status"]
    assert check(problem(4, changed))["status"] == baseline
    assert check(problem(4, [list(map(list, zip(*a))) for a in changed]))["status"] == baseline
