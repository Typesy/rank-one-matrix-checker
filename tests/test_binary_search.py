from itertools import product
import json

import pytest

from rankone.binary_search import binary_maximum, binary_upper_bound, extend_span, outer_bits, unpack_bits
from rankone.cli import main
from rankone.field import Field
from rankone.reference import reference_check
from rankone.binary_membership import binary_membership_check


def test_all_binary_three_by_three_rank_one_candidates_against_library():
    f = Field(2)
    candidates = {outer_bits(u, v, 3, 3) for u in range(1, 8) for v in range(1, 8)}
    assert len(candidates) == 49
    for bits in range(512):
        a = f.GF(unpack_bits(bits, 3, 3)).row_reduce()
        rank = sum(any(int(x) for x in row) for row in a)
        assert (bits in candidates) == (rank == 1)


def test_coset_extension_rejects_dependence_and_new_rank_one():
    ones = {outer_bits(u, v, 2, 2) for u in range(1, 4) for v in range(1, 4)}
    span, _ = extend_span((0,), 1, ones)
    assert extend_span(span, 1, ones) == (None, "dependent")
    assert extend_span(span, 2, ones) == (None, "extra_rank_one")
    extended, reason = extend_span(span, 8, ones)
    assert reason is None and set(extended) == {0, 1, 8, 9}


@pytest.mark.parametrize("m,n", product(range(1, 4), repeat=2))
def test_tiny_maxima_reach_proved_upper_bound_and_reference_validates(m, n):
    expected = 1 if min(m, n) == 1 else 3 if min(m, n) == 2 else 7
    out = binary_maximum(m, n)
    assert out["status"] == "EXACT"
    assert out["reason"] == "proved_upper_bound_attained"
    assert out["lower_bound"] == out["upper_bound"] == expected
    assert reference_check(out["witness"])["status"] == "VALID"
    span_out = binary_maximum(m, n, method="span")
    assert span_out["status"] == "EXACT"
    assert span_out["factor_pairs_bits"] == out["factor_pairs_bits"]
    assert span_out["stats"]["extensions_tested"] == out["stats"]["extensions_tested"]


@pytest.mark.parametrize("limit,reason", [({"max_nodes": 0}, "node_limit"),
                                          ({"max_candidates": 48}, "candidate_limit"),
                                          ({"max_seconds": 0}, "time_limit"),
                                          ({"max_span_size": 1, "method": "span"}, "span_size_limit")])
def test_limits_never_claim_exact(limit, reason):
    out = binary_maximum(3, 3, **limit)
    assert out["status"] == "INCONCLUSIVE" and out["reason"] == reason
    assert out["lower_bound"] == 0 and out["upper_bound"] == 7


def test_partial_search_returns_real_lower_bound_not_maximum():
    out = binary_maximum(3, 3, max_nodes=3)
    assert out["status"] == "INCONCLUSIVE"
    assert 1 <= out["lower_bound"] < out["upper_bound"]
    assert reference_check(out["witness"])["status"] == "VALID"


def test_annihilator_counting_upper_bounds_are_not_claimed_sharp():
    assert binary_upper_bound(4, 4) == 12
    assert binary_upper_bound(5, 5) == 20
    assert binary_upper_bound(3, 6) == binary_upper_bound(6, 3) == 7
    with pytest.raises(ValueError):
        binary_upper_bound(2, 0)


@pytest.mark.parametrize("m,n,kwargs", [(0, 2, {}), (True, 2, {}), (2, 1.0, {}),
                                        (2, 2, {"max_nodes": -1}),
                                        (2, 2, {"max_seconds": float("nan")})])
def test_invalid_search_arguments(m, n, kwargs):
    with pytest.raises(ValueError):
        binary_maximum(m, n, **kwargs)


def test_cli_binary_search_and_saved_witness(tmp_path, capsys):
    output = tmp_path / "maximum.json"
    assert main(["binary-maximum", "3", "3", "--output", str(output)]) == 0
    out = json.loads(output.read_text())
    assert out["lower_bound"] == 7 and out["status"] == "EXACT"
    assert main(["binary-maximum", "3", "3", "--max-nodes", "0"]) == 2
    capsys.readouterr()


@pytest.mark.parametrize("m,n,expected", [(4, 4, 12), (4, 5, 15), (5, 5, 20)])
def test_larger_witnesses_meet_upper_bound_and_independent_ambient_verifier(m, n, expected):
    out = binary_maximum(m, n)
    assert out["status"] == "EXACT" and out["lower_bound"] == out["upper_bound"] == expected
    checked = binary_membership_check(out["witness"])
    assert checked["status"] == "VALID" and checked["dimension"] == expected
    assert len(checked["rank_one_members"]) == expected
    assert checked["candidates_examined"] == ((1 << m) - 1) * ((1 << n) - 1)


@pytest.mark.parametrize("budget", [0, 1, 8, 20, 35, 50])
def test_quotient_and_full_span_search_agree_under_partial_budgets(budget):
    a = binary_maximum(4, 4, max_nodes=budget, method="span")
    b = binary_maximum(4, 4, max_nodes=budget, method="quotient")
    for key in ("status", "reason", "lower_bound", "upper_bound", "factor_pairs_bits"):
        assert a[key] == b[key]
    assert a["stats"]["extensions_tested"] == b["stats"]["extensions_tested"]
