from copy import deepcopy
from itertools import combinations, product
import json
from pathlib import Path
import random

import pytest

from rankone.field import Field
from rankone.fibre import fibre_check, normalize, projective_vectors, verify_fibre
from rankone.linalg import flatten, rank, rank_one
from rankone.problem import prepare
from rankone.reference import reference_check


E11 = [[1, 0], [0, 0]]
E22 = [[0, 0], [0, 1]]
J = [[1, 1], [1, 1]]


def problem(q, matrices, **kwargs):
    return {"field": {"order": q}, "generators": matrices, **kwargs}


def outer(u, v, f):
    return [[f.mul(x, y) for y in v] for x in u]


def ambient_oracle(instance):
    """Full ambient membership, independent of annihilator/fibre construction."""
    f, matrices, canonical = prepare(instance)
    m = len(matrices[0]) if matrices else canonical["rows"]
    n = len(matrices[0][0]) if matrices else canonical["columns"]
    if any(not rank_one(a, f) for a in matrices):
        return "INVALID"
    flat = [flatten(a) for a in matrices]
    if rank(flat, f) != len(matrices):
        return "INVALID"
    permitted = {normalize(row, f) for row in flat}
    for u, v in product(projective_vectors(f, m), projective_vectors(f, n)):
        candidate = flatten(outer(u, v, f))
        if rank(flat + [candidate], f) == len(matrices) and normalize(candidate, f) not in permitted:
            return "INVALID"
    return "VALID"


@pytest.mark.parametrize("q", [2, 3, 4, 5])
def test_all_tiny_projective_families_match_complete_ambient_oracle(q):
    f = Field(q)
    candidates = [outer(u, v, f) for u, v in
                  product(projective_vectors(f, 2), repeat=2)]
    # Exhaust every pair over all four fields; all triples over F2 and F3.
    sizes = (1, 2, 3) if q <= 3 else (1, 2)
    for size in sizes:
        for family in combinations(candidates, size):
            instance = problem(q, list(family))
            report = fibre_check(instance)
            assert report["status"] == ambient_oracle(instance)
            assert verify_fibre(json.loads(json.dumps(report)))["verified"] is True


@pytest.mark.parametrize("q", [2, 3, 4, 5])
def test_fresh_rectangular_families_match_full_coefficients(q):
    f, rng = Field(q), random.Random(2626+q)
    for m,n in ((2,3),(3,2),(3,3)):
        for _ in range(30):
            family = []
            for _ in range(rng.randrange(1,5)):
                u = [rng.randrange(q) for _ in range(m)]
                v = [rng.randrange(q) for _ in range(n)]
                family.append(outer(u, v, f))
            instance = problem(q, family)
            report = fibre_check(instance)
            assert report["status"] == reference_check(instance, max_vectors=100000)["status"]
            assert verify_fibre(report)["verified"] is True


def test_empty_full_and_one_row_edges():
    for q,m,n in ((2,1,1),(4,2,3),(5,4,2)):
        report = fibre_check(problem(q, [], rows=m, columns=n))
        assert report["status"] == "VALID"
        assert report["stats"]["rank_one_directions"] == 0
        assert verify_fibre(report)["verified"] is True
    full = fibre_check(problem(4, [[[2]]]))
    assert full["certificate"]["parity_checks"] == []
    assert verify_fibre(full)["verified"] is True
    for family in ([[[1,2,3]]], [[[1],[2],[3]]]):
        report = fibre_check(problem(4,family))
        assert report["status"] == "VALID"
        assert verify_fibre(report)["verified"] is True
    report = fibre_check(problem(2, [[[1,0]], [[0,1]]]))
    assert report["status"] == "INVALID"
    assert report["certificate"]["matrix"] == [[1,1]]
    assert verify_fibre(report)["verified"] is True


@pytest.mark.parametrize("invalid", [
    problem(2, []), problem(2, [], rows=0, columns=3),
    problem(2, [], rows=True, columns=3),
    problem(2, [E11], rows=3, columns=2),
    problem(4, [[[4]]]), problem(4, [[[True]]]),
])
def test_invalid_shapes_and_labels(invalid):
    with pytest.raises(ValueError):
        fibre_check(invalid)


def test_f4_encoding_and_field_extension_change_the_answer():
    binary = fibre_check(problem(2, [E11,E22,J]))
    extension = fibre_check(problem(4, [E11,E22,J]))
    assert binary["status"] == "VALID"
    assert extension["status"] == "INVALID"
    assert any(x in (2,3) for x in extension["certificate"]["coefficients"])
    assert extension["problem"]["field"]["irreducible_poly"] == [1,1,1]
    assert verify_fibre(extension)["verified"] is True


def test_shared_factor_pair_avoids_returning_an_existing_generator():
    # There are two selected lines in the same fibre. The witness must be their
    # sum, not merely a vector outside the first selected line.
    report = fibre_check(problem(2, [E11, [[0,1],[0,0]]]))
    assert report["reason"] == "extra_rank_one"
    assert report["certificate"]["matrix"] == [[1,1],[0,0]]
    assert report["certificate"]["coefficients"] == [1,1]
    assert verify_fibre(report)["verified"] is True


def test_positive_fibre_certificate_tampering_is_rejected():
    original = fibre_check(problem(3,[E11,E22]))
    assert original["status"] == "VALID"
    for change in ("missing_parity", "dependent_parity", "false_parity", "missing_fibre",
                   "duplicate_fibre", "wrong_left", "nullity", "false_kernel", "count",
                   "stats", "orientation", "shape", "field", "reason", "method"):
        report = deepcopy(original)
        cert = report["certificate"]
        if change == "missing_parity": cert["parity_checks"].pop()
        elif change == "dependent_parity": cert["parity_checks"][1] = cert["parity_checks"][0]
        elif change == "false_parity": cert["parity_checks"][0][0][0] = 1
        elif change == "missing_fibre": cert["fibres"].pop()
        elif change == "duplicate_fibre": cert["fibres"][1] = cert["fibres"][0]
        elif change == "wrong_left": cert["fibres"][0]["left"] = [0,0]
        elif change == "nullity": cert["fibres"][0]["nullity"] = True
        elif change == "false_kernel":
            record = next(x for x in cert["fibres"] if x["nullity"])
            record["kernel_basis"] = [[0,0]]
        elif change == "count": cert["rank_one_directions"] += 1
        elif change == "stats": report["stats"]["fibres_checked"] += 1
        elif change == "orientation": cert["transposed"] = True
        elif change == "shape": cert["shape"] = [1,4]
        elif change == "field": report["problem"]["field"]["characteristic"] = 5
        elif change == "reason": report["reason"] = "other"
        else: report["method"] = "other"
        assert verify_fibre(report)["verified"] is False,change


def test_negative_certificates_and_damaged_witnesses():
    for family in ([E11,E11], [[[0,0],[0,0]]], [[[1,0],[0,1]]], [E11,E22,J]):
        report = fibre_check(problem(3,family))
        assert report["status"] == "INVALID"
        assert verify_fibre(report)["verified"] is True
        damaged = deepcopy(report)
        if "coefficients" in damaged["certificate"]:
            damaged["certificate"]["coefficients"] = [0]*len(family)
        else: damaged["certificate"]["rank"] = 1
        assert verify_fibre(damaged)["verified"] is False
    report = fibre_check(problem(3,[E11,E22,J]))
    for key in ("matrix","u","v"):
        damaged=deepcopy(report)
        if key == "matrix": damaged["certificate"][key][0][0] = 0
        else: damaged["certificate"][key] = [0,0]
        assert verify_fibre(damaged)["verified"] is False


def test_work_time_and_replay_limits_never_assert_a_decision():
    instance=problem(3,[E11,E22])
    for kwargs in ({"max_fibres":0},{"max_fibres":2},{"max_seconds":0}):
        report=fibre_check(instance,**kwargs)
        assert report["status"] == "INCONCLUSIVE"
        assert verify_fibre(report)["verified"] is None
    report=fibre_check(instance)
    assert verify_fibre(report,max_fibres=0)["verified"] is None
    for kwargs in ({"max_fibres":-1},{"max_fibres":True},{"max_seconds":float('nan')},
                   {"max_seconds":float('inf')},{"max_seconds":True}):
        with pytest.raises(ValueError): fibre_check(instance,**kwargs)
    assert verify_fibre({})["verified"] is False


def test_all_research_probe_witnesses_pass_complete_independent_replay():
    from rankone.research import load_witness
    cases=[(q,m,n) for q in (3,4,5) for m,n in
           ((3,3),(3,4),(3,5),(4,4),(4,5),(5,5))]
    assert len(cases)==18
    for q,m,n in cases:
        witness=load_witness(q,m,n)
        report=fibre_check(witness)
        assert report["status"] == "VALID",(q,m,n)
        assert report["stats"]["rank_one_directions"] == len(witness["generators"])
        assert verify_fibre(report)["verified"] is True,(q,m,n)


def test_noncanonical_scalar_representatives_and_padding():
    f=Field(4)
    base=[E11,E22]
    family=[[[f.mul(2,x) for x in row]+[0] for row in a]+[[0,0,0]] for a in base]
    report=fibre_check(problem(4,family))
    assert report["status"] == "VALID"
    assert verify_fibre(report)["verified"] is True
