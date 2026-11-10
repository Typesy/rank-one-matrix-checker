"""Replay the full saved claim, not just its witness or a cached success flag."""

from copy import deepcopy
import time

import pytest

from rankone.certificates import verify_saved
from rankone.constructions import construct_two_row, construct_vandermonde
from rankone.equivalence import canonical_form, classify_families, equivalent
from rankone.fibre import fibre_check
from rankone.operations import execute
from rankone.research import known_classes, known_results
from rankone.search import maximum_search


def example(q=2):
    return construct_two_row(q, 3, k=2)


@pytest.mark.parametrize('method', ['fibre', 'hybrid', 'normalized', 'reference', 'binary_ambient'])
def test_current_and_legacy_check_reports_and_bundles(method):
    saved = execute({'operation': 'check', 'problem': example(), 'method': method})
    assert verify_saved(saved)['verified'] is True
    assert verify_saved(saved['report'])['verified'] is True
    bad = deepcopy(saved)
    bad['problem']['generators'][0][0][0] ^= 1
    assert verify_saved(bad)['verified'] is False
    bad = deepcopy(saved)
    bad['status'] = 'INVALID'
    assert verify_saved(bad)['verified'] is False


@pytest.mark.parametrize('q', [2, 3, 4, 5])
def test_known_bounds_recompute_witness_and_upper_proof(q):
    saved = known_results(q, 3, 3)
    assert verify_saved(saved)['verified'] is True
    for mutation in ('gap', 'upper', 'witness', 'field', 'proof'):
        bad = deepcopy(saved)
        if mutation == 'gap': bad['gap'] += 1
        elif mutation == 'upper': bad['upper_bound'] -= 1
        elif mutation == 'witness': bad['witness']['generators'][0][0][0] = 0 if bad['witness']['generators'][0][0][0] else 1
        elif mutation == 'field': bad['q'] = 3 if q != 3 else 5
        else: bad['upper_bound_proof']['certificate']['left_directions'] += 1
        assert verify_saved(bad)['verified'] is False, mutation


def test_bounded_construction_never_becomes_exact_by_relabeling():
    saved = known_results(3, 4, 4)
    assert saved['status'] == 'BOUNDED'
    assert verify_saved(saved)['verified'] is True
    bad = deepcopy(saved)
    bad['status'] = 'EXACT'
    assert verify_saved(bad)['verified'] is False


@pytest.mark.parametrize('kind,params', [
    ('verified', {'q': 3, 'rows': 3, 'columns': 3}),
    ('two_row', {'q': 4, 'rows': 2, 'columns': 4, 'k': 5}),
    ('vandermonde', {'q': 5, 'rows': 2, 'columns': 2, 'k': 3}),
])
def test_constructor_bundle_input_identity_and_range(kind, params):
    saved = execute({'operation': 'construct', 'construction': kind, **params})
    assert verify_saved(saved)['verified'] is True
    bad = deepcopy(saved)
    bad['report']['problem'] = example(params['q'])
    assert verify_saved(bad)['verified'] is False
    if kind == 'vandermonde':
        report = deepcopy(saved['report'])
        assert verify_saved(report)['verified'] is True
        report['range']['finite_maximum'] += 1
        assert verify_saved(report)['verified'] is False


def test_search_replays_attached_witness_and_mathematical_upper_bound():
    saved = maximum_search(4, 2, 4)
    assert saved['status'] == 'EXACT'
    assert verify_saved(saved)['verified'] is True
    broken = deepcopy(saved)
    broken['witness_check']['problem']['generators'][0][0][0] ^= 1
    assert verify_saved(broken)['verified'] is False
    broken = deepcopy(saved)
    broken['upper_bound_proof']['certificate']['q'] = 99
    assert verify_saved(broken)['verified'] is False
    # A cached success flag is irrelevant when the actual attached evidence is false.
    broken = deepcopy(saved)
    broken['witness_replay']['verified'] = True
    broken['witness_check']['certificate']['rank_one_directions'] += 1
    assert verify_saved(broken)['verified'] is False


def test_false_exhaustive_maximum_cannot_hide_behind_a_valid_witness():
    witness = example()
    saved = {'schema_version': 1, 'method': 'projective_quotient_exhaustive',
             'field': witness['field'], 'q': 2, 'rows': 2, 'columns': 3,
             'status': 'EXACT', 'reason': 'search_exhausted', 'lower_bound': 2,
             'upper_bound': 2, 'search_complete': True, 'maximum_proved': True,
             'upper_bound_proof': {'proof': 'complete_increasing_prefix_search', 'complete': True},
             'witness': witness, 'witness_check': fibre_check(witness)}
    assert verify_saved(saved, max_candidates=10000)['verified'] is False
    greedy = deepcopy(saved)
    greedy['method'] = 'projective_quotient_greedy'
    assert verify_saved(greedy)['verified'] is False
    greedy['search_complete'] = False
    greedy['reason'] = 'greedy_trials_completed'
    assert verify_saved(greedy)['verified'] is False
    assert verify_saved(saved, max_candidates=0)['verified'] is None


def test_legacy_binary_search_and_incomplete_search():
    from rankone.binary_search import binary_maximum
    assert verify_saved(binary_maximum(2, 2))['verified'] is True
    unfinished = maximum_search(3, 4, 4, max_nodes=0)
    assert unfinished['status'] == 'INCONCLUSIVE'
    assert verify_saved(unfinished)['verified'] is None


def test_saved_collection_replays_original_inputs_and_assignments():
    p = example()
    q = deepcopy(p)
    q['generators'].reverse()
    saved = classify_families([p, q], domain_complete=True)
    assert saved['problems'] == [p, q]
    assert verify_saved(saved)['verified'] is True
    partial_scope = classify_families([p, q])
    assert verify_saved(partial_scope)['verified'] is True
    for mutation in ('key', 'assignment', 'input', 'count', 'work', 'complete'):
        bad = deepcopy(saved)
        if mutation == 'key': bad['classes'][0]['canonical_key'] += 'fake'
        elif mutation == 'assignment': bad['classes'][0]['input_indices'] = [0]
        elif mutation == 'input': bad['problems'][0]['generators'][0][0][0] ^= 1
        elif mutation == 'count': bad['class_count'] += 1
        elif mutation == 'work': bad['stats']['candidates_checked'] += 1
        else: bad['all_supplied_inputs_processed'] = False
        assert verify_saved(bad)['verified'] is False, mutation
    old = deepcopy(saved)
    del old['problems']
    assert verify_saved(old)['verified'] is None


@pytest.mark.parametrize('k', [None, 0, 3, 5])
def test_known_classification_theorems_are_recomputed(k):
    saved = known_classes(2, 2, 3, k)
    assert verify_saved(saved)['verified'] is True
    bad = deepcopy(saved)
    if 'class_count' in bad: bad['class_count'] += 1
    else: bad['sizes'].pop()
    assert verify_saved(bad)['verified'] is False


def test_parameter_classification_complete_domain_is_replayed():
    from rankone.classification import classify_parameter_space
    saved = classify_parameter_space(2, 2, 2, 2, max_candidates=10000)
    assert saved['status'] == 'COMPLETE_CLASSIFICATION'
    assert verify_saved(saved, max_candidates=10000)['verified'] is True
    for mutation in ('count', 'coverage', 'work', 'representative'):
        bad = deepcopy(saved)
        if mutation == 'count': bad['class_count'] += 1
        elif mutation == 'coverage': bad['coverage']['all_ambient_rows_retained_at_every_prefix'] = False
        elif mutation == 'work': bad['stats']['good_families_encountered'] += 1
        else: bad['classes'][0]['representative']['generators'][0][0][0] ^= 1
        assert verify_saved(bad, max_candidates=10000)['verified'] is False
    assert verify_saved(saved, max_candidates=0)['verified'] is None


def test_equivalence_and_canonical_reports_dispatch_and_tamper():
    p = example(4)
    for saved in (equivalent(p, p), canonical_form(p)):
        assert verify_saved(saved)['verified'] is True
        bad = deepcopy(saved)
        cert = bad['certificate'] if 'certificate' in bad else bad['transporter']
        cert['scalars'][0] = 0
        assert verify_saved(bad)['verified'] is False


def test_distribution_factor_analysis_and_table_coverage():
    for operation in ('distribution', 'factor_analysis'):
        saved = execute({'operation': operation, 'problem': example(3)})
        assert verify_saved(saved)['verified'] is True
    table = execute({'operation': 'table', 'q': 2})
    assert verify_saved(table)['verified'] is True
    table['report']['entries'].pop()
    assert verify_saved(table)['verified'] is False


def test_wall_time_worker_and_unknown_records():
    saved = fibre_check(example())
    assert verify_saved(saved, max_seconds=5)['verified'] is True
    started = time.monotonic()
    assert verify_saved(saved, max_seconds=0.001)['verified'] is None
    assert time.monotonic()-started < 2
    assert verify_saved(saved, max_seconds=0)['verified'] is None
    assert verify_saved({'status': 'EXACT', 'cached_verified': True})['verified'] is None
    assert verify_saved({'status': 'COMPLETE', 'entries': [], 'scope': 'unrelated table'})['verified'] is None
    assert verify_saved({'operation': 'replay', 'report': {'status': 'VERIFIED', 'replay': {'verified': True}},
                         'status': 'VERIFIED'})['verified'] is None
    nested = saved
    for _ in range(40):
        nested = {'operation': 'check', 'report': nested, 'status': 'VALID'}
    assert verify_saved(nested)['verified'] is False
    cyclic = {'operation': 'check', 'status': 'VALID'}
    cyclic['report'] = cyclic
    assert verify_saved(cyclic)['verified'] is False
    assert verify_saved(cyclic, max_seconds=2)['verified'] is False
    bad = known_classes(2, 2, 2, 1)
    bad['class_count'] = True
    assert verify_saved(bad)['verified'] is False
    mislabeled = execute({'operation': 'check', 'problem': example()})
    mislabeled['operation'] = 'search'
    assert verify_saved(mislabeled)['verified'] is False
    for kwargs in ({'max_seconds': -1}, {'max_candidates': True}, {'max_seconds': float('nan')}):
        with pytest.raises(ValueError): verify_saved(saved, **kwargs)


@pytest.mark.parametrize('method_mutation', ['unknown', 'missing'])
def test_damaged_checker_cannot_use_constructor_replay_as_a_bypass(method_mutation):
    report = fibre_check(example(3))
    report['certificate']['rank_one_directions'] = 999
    if method_mutation == 'unknown':
        report['method'] = 'unknown_method'
    else:
        del report['method']
    assert verify_saved(report)['verified'] is False
    # Wrapping the same record as a constructor is rejected too, even when
    # the attached ordinary check independently certifies the underlying set.
    good_check = execute({'operation': 'check', 'problem': example(3)})
    wrapper = {'operation': 'construct', 'status': 'VALID', 'report': report,
               'problem': report['problem'], 'check': good_check,
               'construction_status': 'VALID'}
    assert verify_saved(wrapper)['verified'] is False
