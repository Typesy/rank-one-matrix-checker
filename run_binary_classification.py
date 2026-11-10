#!/usr/bin/env python3
"""Exhaustive classification of extremal 3x3 binary rank-one basis spaces.

Run from research_checker:
    .venv/bin/python run_binary_classification.py \
        --output results/binary_3x3_classification.json

This calculation is independent of rankone.checker, binary_search, and galois.
Its positive and negative decisions use Gaussian elimination over F_2 and
membership tests for *all* 49 ambient rank-one matrices. It is reproducible
finite computation, not formal verification or a claim of mathematical novelty.
"""

import argparse
from collections import Counter
from itertools import permutations
import json
from math import factorial
from pathlib import Path

FACTORS = tuple(range(1, 8))
# Integer labels are binary coordinates, lowest bit first; e.g. 5=(1,0,1).
COMPANION_WITNESS = (2, 5, 4, 1, 7, 6, 3)


def outer(u, v):
    """Pack uv^T in row-major bits, with 3 consecutive bits per row."""
    return sum(v << (3 * i) for i in range(3) if (u >> i) & 1)


def unpack(matrix):
    return [[(matrix >> (3 * i + j)) & 1 for j in range(3)] for i in range(3)]


def echelon_basis(vectors):
    """Return a pivot dictionary spanning the input binary vectors."""
    pivots = {}
    for vector in vectors:
        while vector:
            pivot = vector.bit_length() - 1
            if pivot in pivots:
                vector ^= pivots[pivot]
            else:
                pivots[pivot] = vector
                break
    return pivots


def contains(pivots, vector):
    while vector:
        pivot = vector.bit_length() - 1
        if pivot not in pivots:
            return False
        vector ^= pivots[pivot]
    return True


def ambient_rank_one():
    return frozenset(outer(u, v) for u in FACTORS for v in FACTORS)


def inspect_family(right_factors, candidates=None):
    """Test one bijection from the seven nonzero left factors to right factors.

    Independence is separate from the condition that the span contains no
    additional rank-one matrices. Some dependent seven-element families satisfy
    the second condition, but do not define a dimension-seven basis space.
    """
    right_factors = tuple(right_factors)
    if len(right_factors) != 7 or set(right_factors) != set(FACTORS):
        raise ValueError("right_factors must be a permutation of 1,...,7")
    candidates = ambient_rank_one() if candidates is None else candidates
    generators = frozenset(outer(u, v) for u, v in zip(FACTORS, right_factors))
    pivots = echelon_basis(generators)
    intersection = frozenset(a for a in candidates if contains(pivots, a))
    extra = intersection - generators
    return {
        "dimension": len(pivots),
        "rank_one_count": len(intersection),
        "extra_rank_one_count": len(extra),
        "independent": len(pivots) == 7,
        "good": len(pivots) == 7 and not extra,
    }


def invertible_maps():
    """All GL(3,2) maps, represented by their eight vector images.

    Ordered images a,b,c of the standard basis are independent exactly when
    a,b are distinct nonzero and c is outside {0,a,b,a+b}. The permutation loop
    excludes 0,a,b, so only c=a+b still has to be excluded.
    """
    maps = []
    for a, b, c in permutations(FACTORS, 3):
        if c == a ^ b:
            continue
        maps.append(tuple(
            (a if x & 1 else 0) ^ (b if x & 2 else 0) ^ (c if x & 4 else 0)
            for x in range(8)
        ))
    return tuple(maps)


def transform_family(right_factors, left_map, right_map):
    """Action on factor pairs: (u,v) -> (P u, R v).

    This is the matrix action M -> P M R^T. Since transposition permutes GL(3,2),
    the resulting orbit is exactly the requested M -> P M Q equivalence orbit.
    The matrices are an unordered family; sort again by the transformed left
    factor to recover the unique permutation representation.
    """
    transformed = [0] * 7
    for u, v in zip(FACTORS, right_factors):
        transformed[left_map[u] - 1] = right_map[v]
    return tuple(transformed)


def classify():
    candidates = ambient_rank_one()
    good_families = set()
    dimensions = Counter()
    intersections = Counter()
    rejection_types = Counter()
    examined = 0
    for family in permutations(FACTORS):
        inspected = inspect_family(family, candidates)
        examined += 1
        dimensions[inspected["dimension"]] += 1
        intersections[inspected["rank_one_count"]] += 1
        if inspected["good"]:
            good_families.add(family)
        else:
            reason = (
                "dependent_and_extra" if not inspected["independent"] and inspected["extra_rank_one_count"]
                else "dependent_only" if not inspected["independent"]
                else "extra_rank_one_only"
            )
            rejection_types[reason] += 1

    gl = invertible_maps()
    orbit = {
        transform_family(COMPANION_WITNESS, left, right)
        for left in gl for right in gl
    }
    witness_valid = COMPANION_WITNESS in good_families
    complete = examined == factorial(7)
    orbit_equal = orbit == good_families
    group_order = len(gl) ** 2
    return {
        "schema_version": 1,
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "parameters": {"q": 2, "m": 3, "n": 3, "k": 7},
        "encoding": "u,v integers 1..7 give binary coordinates lowest bit first",
        "candidate_rank_one_matrices": len(candidates),
        "full_factor_bijections_examined": examined,
        "expected_full_factor_bijections": factorial(7),
        "good_independent_families": len(good_families),
        "rejections": dict(sorted(rejection_types.items())),
        "dimension_histogram": {str(k): v for k, v in sorted(dimensions.items())},
        "span_rank_one_count_histogram": {str(k): v for k, v in sorted(intersections.items())},
        "GL3_order": len(gl),
        "group_pairs_examined": group_order,
        "witness_orbit_size": len(orbit),
        "witness_is_good": witness_valid,
        "orbit_equals_all_good_families": orbit_equal,
        "extremal_equivalence_classes": 1 if complete and witness_valid and orbit_equal else None,
        "witness_stabilizer_order": group_order // len(orbit),
        "witness": {
            "left_factors": list(FACTORS),
            "right_factors": list(COMPANION_WITNESS),
            "generators": [unpack(outer(u, v)) for u, v in zip(FACTORS, COMPANION_WITNESS)],
        },
        "coverage_argument": [
            "A good independent rank-one basis cannot repeat a left or right nonzero factor: the sum of two such generators would be an additional rank-one matrix.",
            "Over F_2^3 there are seven nonzero factors. Thus k<=7, and a size-seven good basis uses every factor once on both sides.",
            "Binary rank-one matrices have unique nonzero factorizations uv^T. Ordering the left factors 1..7 makes every possible extremal basis exactly one of the 7! right-factor permutations.",
            "For each permutation, Gaussian elimination checks independence and span membership of all 49 ambient rank-one matrices.",
            "Every GL(3,2) map is represented by exactly one ordered independent triple of standard-basis images. All 168 maps on each side are enumerated.",
            "Each good space has exactly its seven basis matrices as rank-one elements, so its unordered rank-one family uniquely determines the space. Family equivalence therefore proves space equivalence in this extremal case.",
        ],
        "scope": "Exact finite computational classification, subject to implementation correctness; no formal-proof-kernel or novelty claim. Transpose equivalence is unnecessary because there is already one left/right orbit.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/binary_3x3_classification.json"))
    args = parser.parse_args()
    report = classify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "status", "full_factor_bijections_examined", "good_independent_families",
        "rejections", "witness_orbit_size", "orbit_equals_all_good_families",
        "extremal_equivalence_classes", "witness_stabilizer_order",
    )}, indent=2))


if __name__ == "__main__":
    main()
