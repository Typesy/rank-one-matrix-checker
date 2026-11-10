from itertools import combinations, product

from rankone.reference import reference_check


def test_no_four_generator_binary_two_by_two_basis():
    # Every binary rank-one 2x2 matrix is uv^T for nonzero u,v in F_2^2.
    vectors = [v for v in product(range(2), repeat=2) if any(v)]
    candidates = [[[x * y for y in v] for x in u] for u in vectors for v in vectors]
    assert len({tuple(sum(a, [])) for a in candidates}) == 9
    tested = 0
    for family in combinations(candidates, 4):
        out = reference_check({"field": {"order": 2}, "generators": list(family)})
        assert out["status"] == "INVALID"
        tested += 1
    assert tested == 126
