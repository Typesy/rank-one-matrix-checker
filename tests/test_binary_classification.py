"""Coverage and independent cross-checks for the small exact classification."""

import importlib.util
from itertools import permutations
from pathlib import Path

import pytest

from rankone.reference import reference_check

SCRIPT = Path(__file__).resolve().parents[1] / "run_binary_classification.py"
spec = importlib.util.spec_from_file_location("binary_classification", SCRIPT)
classification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(classification)


def test_binary_candidate_grid_and_gaussian_membership():
    c = classification
    assert len(c.ambient_rank_one()) == 49
    assert c.outer(5, 6) == 6 + (6 << 6)
    assert c.unpack(c.outer(5, 6)) == [[0, 1, 1], [0, 0, 0], [0, 1, 1]]
    pivots = c.echelon_basis([0b101, 0b110, 0b011])
    assert len(pivots) == 2
    assert all(c.contains(pivots, x) for x in (0, 0b101, 0b110, 0b011))
    assert not c.contains(pivots, 0b001)


def test_dependent_no_extra_set_is_not_good_independent_basis():
    # uu^T for all seven nonzero u: no additional rank-one matrix, but dimension 6.
    inspected = classification.inspect_family(range(1, 8))
    assert inspected == {
        "dimension": 6, "rank_one_count": 7, "extra_rank_one_count": 0,
        "independent": False, "good": False,
    }


def test_decisions_against_existing_reference_on_all_first_factor_fixed_cases():
    # 120 cases are enough to cover independent valid, dependent, and extra-point
    # failures here, without making the full small classification rely on galois.
    c = classification
    seen = set()
    for tail in permutations(range(2, 8)):
        family = (1,) + tail
        # A deterministic 120-case slice with second image2: all 5! tails.
        if family[1] != 2:
            continue
        inspected = c.inspect_family(family)
        reference = reference_check({
            "field": {"order": 2},
            "generators": [c.unpack(c.outer(u, v)) for u, v in zip(c.FACTORS, family)],
        })
        assert inspected["good"] == (reference["status"] == "VALID")
        seen.add(reference["reason"])
    assert seen == {"exhausted_full_coefficients", "dependent_generators", "extra_rank_one"}


def test_gl_maps_exhaustive_and_linear():
    maps = classification.invertible_maps()
    assert len(set(maps)) == 168
    for mapping in maps:
        assert mapping[0] == 0 and set(mapping) == set(range(8))
        assert all(mapping[a ^ b] == mapping[a] ^ mapping[b]
                   for a in range(8) for b in range(8))


def test_complete_classification_and_orbit_coverage():
    report = classification.classify()
    assert report["status"] == "COMPLETE"
    assert report["full_factor_bijections_examined"] == 5040
    assert report["good_independent_families"] == 1344
    assert report["rejections"] == {"dependent_only": 1344, "extra_rank_one_only": 2352}
    assert report["dimension_histogram"] == {"6": 1344, "7": 3696}
    assert report["GL3_order"] == 168
    assert report["group_pairs_examined"] == 28224
    assert report["witness_orbit_size"] == 1344
    assert report["witness_is_good"] and report["orbit_equals_all_good_families"]
    assert report["extremal_equivalence_classes"] == 1
    assert report["witness_stabilizer_order"] == 21


@pytest.mark.parametrize("bad", [(1, 2), (1, 1, 2, 3, 4, 5, 6), (0, 1, 2, 3, 4, 5, 6)])
def test_rejects_non_bijections(bad):
    with pytest.raises(ValueError):
        classification.inspect_family(bad)
