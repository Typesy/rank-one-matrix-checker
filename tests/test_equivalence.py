"""Independent small group oracles, transporter replay, and canonical limits."""

from copy import deepcopy
from itertools import combinations, product
import json
from pathlib import Path
import random

import numpy as np
import pytest

from rankone.equivalence import (
    canonical_form, classify_families, equivalent, small_class_representatives,
    verify_canonical, verify_equivalence,
)
from rankone.field import Field


def family(q, left, right, rows=None, columns=None):
    f = Field(q)
    m, n = rows or len(left[0]), columns or len(right[0])
    matrices = []
    for u, v in zip(left, right):
        u, v = u + [0]*(m-len(u)), v + [0]*(n-len(v))
        matrices.append((f.GF(u)[:, None] * f.GF(v)[None, :]).tolist())
    return {"field": f.metadata(), "generators": matrices}


def transformed(problem, seed=23, *, transpose=False, exponent=0):
    f = Field(problem['field']['order'], problem['field'].get('irreducible_poly'))
    a = f.GF(problem['generators'])
    k, m, n = a.shape
    rng = random.Random(seed)

    def invertible(d):
        while True:
            x = f.GF([[rng.randrange(f.q) for _ in range(d)] for _ in range(d)])
            if np.linalg.matrix_rank(x) == d:
                return x

    p, q = invertible(m), invertible(n)
    order = list(range(k))
    rng.shuffle(order)
    result = []
    for i in order:
        b = a[i] ** (f.p**exponent)
        if transpose:
            b = b.T
        result.append((f.GF(rng.randrange(1, f.q)) * (p @ b @ q)).tolist())
    return {"field": f.metadata(), "generators": result}


@pytest.mark.parametrize('q', [2, 3, 4, 5])
def test_transport_and_canonical_all_fields(q):
    p = family(q, [[1, 0], [0, 1], [1, 1]],
               [[1, 0, 0], [0, 1, 0], [0, 0, 1]], rows=4, columns=5)
    target = transformed(p, seed=q+120)
    report = equivalent(p, target)
    assert report['status'] == 'EQUIVALENT'
    assert verify_equivalence(report)['verified'] is True
    a, b = canonical_form(p), canonical_form(target)
    assert a['status'] == b['status'] == 'COMPLETE'
    assert a['canonical_key'] == b['canonical_key']
    assert verify_canonical(a)['verified'] is True
    assert canonical_form(a['representative'])['canonical_key'] == a['canonical_key']


def test_binary_three_by_three_rank_pair_classes_and_transpose():
    plain = small_class_representatives(2, 3, 3, 3)
    assert plain['status'] == 'COMPLETE_CLASSIFICATION'
    assert plain['class_count'] == 4
    reps = plain['representatives']
    for i, j in combinations(range(4), 2):
        r = equivalent(reps[i], reps[j])
        assert r['status'] == 'INEQUIVALENT'
        assert verify_equivalence(r)['verified'] is True
    assert equivalent(reps[1], reps[2], allow_transpose=True)['status'] == 'EQUIVALENT'
    assert small_class_representatives(2, 3, 3, 3, allow_transpose=True)['class_count'] == 3
    c = [canonical_form(x, allow_transpose=True)['canonical_key'] for x in reps]
    assert c[1] == c[2] and len(set(c)) == 3


@pytest.mark.parametrize('q', [3, 4, 5])
def test_nonbinary_small_classes(q):
    assert small_class_representatives(q, 3, 3, 3)['class_count'] == 3
    assert small_class_representatives(q, 3, 3, 3, allow_transpose=True)['class_count'] == 2
    assert small_class_representatives(q, 2, 2, 3)['class_count'] == 0
    assert small_class_representatives(q, 1, 5, 2)['class_count'] == 0
    assert small_class_representatives(q, 4, 4, 4)['status'] == 'UNAVAILABLE'


def test_same_distribution_two_binary_classes_rejected():
    left = [[1, 0], [0, 1], [1, 1]]
    circuit = family(2, left, left, columns=3)
    independent = family(2, left, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    result = equivalent(circuit, independent)
    assert result['status'] == 'INEQUIVALENT'
    assert result['reason'] == 'essential_support_ranks_differ'
    assert canonical_form(circuit)['canonical_key'] != canonical_form(independent)['canonical_key']


def test_limits_and_no_partial_canonical_key():
    p = family(3, [[1, 0], [0, 1]], [[1, 0], [0, 1]])
    assert equivalent(p, p, max_candidates=0)['status'] == 'INCONCLUSIVE'
    c = canonical_form(p, max_candidates=0)
    assert c['status'] == 'INCONCLUSIVE'
    assert 'canonical_key' not in c
    assert verify_canonical(c)['verified'] is None
    assert canonical_form(p, max_seconds=0)['status'] == 'INCONCLUSIVE'
    report = classify_families([p, p], max_candidates=0, domain_complete=True)
    assert report['status'] == 'PARTIAL_CLASSIFICATION'
    for kwargs in ({'max_candidates': -1}, {'max_candidates': True}, {'max_seconds': float('nan')}):
        with pytest.raises(ValueError):
            equivalent(p, p, **kwargs)


def test_invalid_basis_and_field_mismatch():
    p = family(3, [[1, 0], [0, 1]], [[1, 0], [0, 1]])
    bad = {'field': {'order': 3}, 'generators': p['generators'] + [p['generators'][0]]}
    assert equivalent(p, bad)['status'] == 'INVALID_INPUT'
    assert canonical_form(bad)['status'] == 'INVALID_INPUT'
    same_left = family(3, [[1, 0], [1, 0]], [[1, 0], [0, 1]])
    assert equivalent(same_left, same_left)['status'] == 'INVALID_INPUT'
    other = deepcopy(p)
    other['field'] = {'order': 5}
    with pytest.raises(ValueError, match='same explicit field'):
        equivalent(p, other)
    # Same order is insufficient when polynomial-basis encodings differ.
    first = deepcopy(p)
    first['field'] = {'order': 8, 'irreducible_poly': [1, 0, 1, 1]}
    second = deepcopy(p)
    second['field'] = {'order': 8, 'irreducible_poly': [1, 1, 0, 1]}
    with pytest.raises(ValueError, match='same explicit field'):
        equivalent(first, second)


def test_positive_certificate_tampering():
    p = family(4, [[1, 0], [0, 1]], [[1, 0], [0, 1]])
    report = equivalent(p, transformed(p))
    assert verify_equivalence(report)['verified'] is True
    for edit in ('singular', 'scalar', 'matching', 'matrix', 'transpose', 'automorphism', 'boolean'):
        bad = deepcopy(report)
        c = bad['certificate']
        if edit == 'singular': c['P'][0] = [0, 0]
        elif edit == 'scalar': c['scalars'][0] = 0
        elif edit == 'matching': c['permutation'] = [0, 0]
        elif edit == 'matrix': bad['target']['generators'][0][0][0] ^= 1
        elif edit == 'transpose': c['transpose'] = True
        elif edit == 'automorphism': c['frobenius_power'] = 1
        else: c['permutation'][0] = False
        assert verify_equivalence(bad)['verified'] is False, edit
    canonical = canonical_form(p)
    canonical['canonical_key'] += 'tamper'
    assert verify_canonical(canonical)['verified'] is False


def test_empty_and_singletons():
    empty = {'field': {'order': 4}, 'rows': 2, 'columns': 3, 'generators': []}
    r = equivalent(empty, empty)
    assert r['status'] == 'EQUIVALENT'
    assert verify_equivalence(r)['verified'] is True
    c = canonical_form(empty)
    assert c['status'] == 'COMPLETE'
    assert verify_canonical(c)['verified'] is True
    for m, n in [(1, 1), (1, 4), (5, 1)]:
        a = small_class_representatives(5, m, n, 1)['representatives'][0]
        b = transformed(a)
        assert equivalent(a, b)['status'] == 'EQUIVALENT'
        assert canonical_form(a)['canonical_key'] == canonical_form(b)['canonical_key']


def test_supplied_domain_classification_is_explicit():
    reps = small_class_representatives(2, 3, 3, 3)['representatives']
    data = reps + [transformed(x, seed=91+i) for i, x in enumerate(reps)]
    out = classify_families(data, domain_complete=True)
    assert out['status'] == 'COMPLETE_CLASSIFICATION'
    assert out['scope'] == 'supplied_family_list_only'
    assert out['class_count'] == 4
    assert sorted(len(c['input_indices']) for c in out['classes']) == [2]*4
    assert classify_families(data)['status'] == 'PARTIAL_CLASSIFICATION'


def test_all_binary_two_by_two_pairs_against_direct_group_oracle():
    gf = Field(2).GF
    nonzero = [(1, 0), (0, 1), (1, 1)]
    rankones = [(gf(u)[:, None]*gf(v)[None, :]).tolist() for u in nonzero for v in nonzero]
    group = []
    for flat in product(range(2), repeat=4):
        a = gf([flat[:2], flat[2:]])
        if np.linalg.matrix_rank(a) == 2:
            group.append(a)
    base = family(2, [[1, 0], [0, 1]], [[1, 0], [0, 1]])
    orbit = set()
    for p in group:
        for q in group:
            orbit.add(tuple(sorted(tuple((p@gf(a)@q).flatten().tolist()) for a in base['generators'])))
    for a, b in combinations(rankones, 2):
        candidate = {'field': {'order': 2}, 'generators': [a, b]}
        expected = tuple(sorted((tuple(sum(a, [])), tuple(sum(b, []))))) in orbit
        result = equivalent(base, candidate)
        assert (result['status'] == 'EQUIVALENT') == expected
        if expected:
            assert verify_equivalence(result)['verified'] is True
        else:
            assert result['status'] == 'INVALID_INPUT'


def test_five_field_same_matroids_inequivalent_fixture():
    points = [[1, t, t*t % 5, t*t*t % 5] for t in range(5)] + [[0, 0, 0, 1]]
    a = family(5, points, points)
    right = list(points)
    right[0], right[1] = right[1], right[0]
    b = family(5, points, right)
    result = equivalent(a, b, max_candidates=30000)
    assert result['status'] == 'INEQUIVALENT'
    assert result['reason'] == 'complete_transporter_search'
    assert verify_equivalence(result, max_candidates=30000)['verified'] is True
    broken = deepcopy(result)
    broken['certificate']['candidates_checked'] -= 1
    assert verify_equivalence(broken, max_candidates=30000)['verified'] is False


def test_frobenius_map_certificate_uses_both_factors():
    a = family(4, [[1, 0], [0, 1], [1, 2]],
               [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    b = transformed(a, exponent=1)
    report = equivalent(a, b, allow_frobenius=True)
    assert report['status'] == 'EQUIVALENT'
    assert verify_equivalence(report)['verified'] is True
    assert canonical_form(a, allow_frobenius=True)['canonical_key'] == canonical_form(b, allow_frobenius=True)['canonical_key']


def test_frobenius_really_merges_two_distinct_linear_orbits():
    left = [[1, 0, 0, 3], [1, 0, 1, 2], [0, 1, 1, 0], [1, 3, 3, 2],
            [1, 3, 0, 3], [1, 3, 1, 2], [1, 2, 3, 0]]
    right = [[1, 1, 0, 3], [0, 1, 0, 1], [1, 3, 3, 2], [1, 2, 2, 1],
             [1, 3, 1, 1], [1, 0, 3, 0], [1, 1, 1, 3]]
    source = family(4, left, right)
    gf = Field(4).GF
    target = {'field': source['field'],
              'generators': [(gf(a)**2).tolist() for a in source['generators']]}
    assert equivalent(source, target, max_candidates=30000)['status'] == 'INEQUIVALENT'
    out = equivalent(source, target, allow_frobenius=True, max_candidates=60000)
    assert out['status'] == 'EQUIVALENT'
    assert out['certificate']['frobenius_power'] == 1
    assert verify_equivalence(out)['verified'] is True
    wrong = deepcopy(out)
    wrong['certificate']['frobenius_power'] = 0
    assert verify_equivalence(wrong)['verified'] is False
    ca = canonical_form(source, allow_frobenius=True, max_candidates=60000)
    cb = canonical_form(target, allow_frobenius=True, max_candidates=60000)
    assert ca['status'] == cb['status'] == 'COMPLETE'
    assert ca['canonical_key'] == cb['canonical_key']


def test_ordered_right_normalization_against_all_small_projectivities():
    from rankone.equivalence import _ordered_right_canonical
    f = Field(3)
    points = [[1, 0], [0, 1], [1, 1], [1, 2]]
    expected, _ = _ordered_right_canonical(points, f)
    count = 0
    for flat in product(range(3), repeat=4):
        matrix = f.GF([flat[:2], flat[2:]])
        if np.linalg.matrix_rank(matrix) != 2:
            continue
        # Independent column rescalings too: they cannot change the ordered object.
        changed = [(f.GF(1+i%2) * (matrix @ f.GF(p))).tolist() for i, p in enumerate(points)]
        actual, _ = _ordered_right_canonical(changed, f)
        assert actual == expected
        count += 1
    assert count == 48


def test_transpose_does_not_act_inside_a_rectangular_ambient_shape():
    two = [[1, 0], [0, 1], [1, 1]]
    three = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    a = family(3, two, three, rows=3, columns=4)
    b = family(3, three, two, rows=3, columns=4)
    out = equivalent(a, b, allow_transpose=True)
    assert out['status'] == 'INEQUIVALENT'
    assert verify_equivalence(out)['verified'] is True
    bad = deepcopy(out)
    bad['certificate']['target_ranks'] = [4, 4]
    assert verify_equivalence(bad)['verified'] is False
    good = equivalent(a, a)
    good['action']['generator_order'] = 'ordered'
    assert verify_equivalence(good)['verified'] is False
