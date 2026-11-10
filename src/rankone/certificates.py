"""Replay saved research results, including the mathematical claims in bundles.

Cached booleans are never accepted as evidence. Complete search/classification
claims need matching proofs or a completed replay. Supplying max_seconds uses
one bounded subprocess, including legacy verifiers without internal timeouts.
"""

import json
from math import isfinite
import subprocess
import sys
from time import monotonic


class _Limit(Exception):
    pass


class _Budget:
    def __init__(self, count, seconds):
        if type(count) is not int or count < 0:
            raise ValueError("max_candidates must be a nonnegative integer")
        if seconds is not None and (type(seconds) not in (int, float)
                                    or not isfinite(seconds) or seconds < 0):
            raise ValueError("max_seconds must be finite and nonnegative")
        self.count, self.seconds, self.started = count, seconds, monotonic()
        self.depth = 0

    def remaining(self):
        return None if self.seconds is None else max(0, self.seconds-(monotonic()-self.started))

    def check(self):
        if self.remaining() == 0:
            raise _Limit


def _answer(verified, reason, **extra):
    return {"verified": verified, "reason": reason, **extra}


def _require(condition, reason="inconsistent_saved_result"):
    if not condition:
        raise ValueError(reason)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _equal(a, b):
    """JSON equality with booleans distinct from integer proof counts."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def _semantic_problem(problem):
    from .problem import prepare
    _, a, p = prepare(problem)
    shape = [len(a[0]), len(a[0][0])] if a else [p['rows'], p['columns']]
    return {"field": p['field'], "shape": shape, "generators": a}


def _same_problem(a, b):
    return _semantic_problem(a) == _semantic_problem(b)


def _parameters(report):
    from .field import Field
    q = report.get('q', report.get('field_order'))
    m, n = report['rows'], report['columns']
    _require(_integer(q, 2) and _integer(m, 1) and _integer(n, 1))
    spec = report.get('field', {'order': q})
    _require(isinstance(spec, dict) and spec.get('order') == q)
    f = Field(q, spec.get('irreducible_poly'))
    metadata = f.metadata()
    for key in ('characteristic', 'degree', 'encoding'):
        _require(key not in spec or spec[key] == metadata[key], 'field_model_mismatch')
    return q, m, n, metadata


def _witness(report, budget):
    from .fibre import fibre_check, verify_fibre
    q, m, n, field = _parameters(report)
    witness = report['witness']
    # Historical binary empty witnesses predate explicit zero-space shape.
    if witness.get('generators') == [] and 'rows' not in witness:
        witness = {**witness, 'rows': m, 'columns': n}
    semantic = _semantic_problem(witness)
    _require(semantic['field'] == field and semantic['shape'] == [m, n], 'witness_parameters_disagree')
    lower = report['lower_bound']
    _require(_integer(lower) and lower == len(semantic['generators']), 'witness_size_disagrees')
    attached = report.get('witness_check')
    if attached is not None:
        _require(_same_problem(attached['problem'], witness), 'attached_check_uses_a_different_witness')
        replay = _dispatch(attached, budget)
        if replay['verified'] is not True:
            return replay
        _require(attached['status'] == 'VALID', 'witness_check_does_not_certify_validity')
    else:
        budget.check()
        checked = fibre_check(witness, max_fibres=budget.count, max_seconds=budget.remaining())
        if checked['status'] == 'INCONCLUSIVE':
            return _answer(None, 'witness_replay_limit')
        _require(checked['status'] == 'VALID', 'saved_witness_is_not_a_good_basis')
        replay = verify_fibre(checked, max_fibres=budget.count)
        if replay['verified'] is not True:
            return replay
    budget.check()
    return _answer(True, 'witness_validity_replayed')


def _bounds(report, budget, *, search=False):
    from .research import upper_bound
    q, m, n, field = _parameters(report)
    lower, upper = report['lower_bound'], report['upper_bound']
    _require(_integer(lower) and _integer(upper) and lower <= upper <= m*n)
    status = report['status']
    _require(status in ('EXACT', 'BOUNDED', 'INCONCLUSIVE'))
    _require(status != 'EXACT' or lower == upper, 'exact_claim_has_a_bound_gap')
    if 'gap' in report:
        _require(type(report['gap']) is int and report['gap'] == upper-lower)
    if 'maximum_proved' in report:
        _require(type(report['maximum_proved']) is bool and report['maximum_proved'] == (status == 'EXACT'))
    if 'search_complete' in report:
        _require(type(report['search_complete']) is bool)
        _require(not report['search_complete'] or report['method'].endswith('_exhaustive'),
                 'greedy_search_cannot_claim_exhaustion')
    verified = _witness(report, budget)
    if verified['verified'] is not True:
        return verified
    mathematical = upper_bound(q, m, n, field_spec=field)
    fresh_upper = mathematical['upper_bound']
    proof = report.get('upper_bound_proof')
    exhausted = search and (report.get('search_complete') is True or report.get('reason') == 'search_exhausted')
    if exhausted:
        _require(status == 'EXACT' and lower == upper, 'exhausted_search_cannot_have_an_open_gap')
    if proof is not None and proof.get('proof') == 'complete_increasing_prefix_search':
        _require(exhausted, 'complete_search_proof_has_no_completed_search_claim')
    if proof is not None and proof.get('proof') != 'complete_increasing_prefix_search':
        _require(_equal(proof, mathematical), 'upper_bound_certificate_disagrees')
    if upper == fresh_upper:
        if status == 'BOUNDED':
            _require(lower < upper, 'bounded_status_has_no_gap')
        return _answer(None if status == 'INCONCLUSIVE' else True,
                       'partial_bounds_replayed' if status == 'INCONCLUSIVE' else
                       'witness_and_mathematical_upper_bound_replayed', claim=status)
    if not search:
        return _answer(False, 'upper_bound_does_not_match_its_proof')
    if not exhausted:
        return _answer(False, 'smaller_upper_bound_has_no_complete_search_proof')
    if proof is not None:
        _require(proof == {'proof': 'complete_increasing_prefix_search', 'complete': True})
    budget.check()
    method = report['method']
    if method.startswith('binary_partial_matchings_'):
        from .binary_search import binary_maximum
        fresh = binary_maximum(m, n, method=method.rsplit('_', 1)[1],
                               max_nodes=budget.count, max_candidates=budget.count,
                               max_span_size=budget.count, max_seconds=budget.remaining())
    else:
        from .search import maximum_search
        fresh = maximum_search(q, m, n, method='exhaustive', use_known=False,
                               max_nodes=budget.count, max_candidates=budget.count,
                               max_seconds=budget.remaining(), field_spec=field)
    if fresh['status'] != 'EXACT':
        return _answer(None, 'exhaustive_search_replay_limit')
    _require(fresh['lower_bound'] == lower and fresh['upper_bound'] == upper,
             'exhaustive_search_disagrees_with_saved_maximum')
    budget.check()
    return _answer(True, 'exact_maximum_reproved_by_complete_search')


def _collection(report, budget):
    from .equivalence import classify_families
    if 'problems' not in report:
        return _answer(None, 'original_collection_inputs_not_recorded')
    problems, action = report['problems'], report['action']
    _require(isinstance(problems, list))
    _require(type(report['declared_domain_complete']) is bool)
    _require(report['scope'] == 'supplied_family_list_only')
    _require(type(report['families_supplied']) is int and report['families_supplied'] == len(problems))
    _require(type(report['class_count']) is int and report['class_count'] == len(report['classes']))
    _require(_integer(report['families_processed']) and report['families_processed'] <= len(problems))
    _require(len(report['records']) == report['families_processed'])
    _require([r['input_index'] for r in report['records']] == list(range(report['families_processed'])))
    expected_complete = (report['families_processed'] == len(problems) and
                         all(r['status'] != 'INCONCLUSIVE' for r in report['records']))
    _require(type(report['all_supplied_inputs_processed']) is bool and
             report['all_supplied_inputs_processed'] == expected_complete)
    status = ('COMPLETE_CLASSIFICATION' if expected_complete and report['declared_domain_complete']
              else 'PARTIAL_CLASSIFICATION')
    _require(report['status'] == status)
    if not expected_complete:
        return _answer(None, 'collection_partition_is_incomplete')
    fresh = classify_families(problems, allow_transpose=action['allow_transpose'],
                              allow_frobenius=action['allow_frobenius'],
                              max_candidates=budget.count, max_seconds=budget.remaining(),
                              domain_complete=report['declared_domain_complete'])
    if fresh['all_supplied_inputs_processed'] is not True:
        return _answer(None, 'collection_replay_limit')
    for key in ('status', 'scope', 'action', 'class_count', 'classes', 'records',
                'families_supplied', 'families_processed', 'all_supplied_inputs_processed'):
        _require(_equal(report[key], fresh[key]), 'collection_partition_or_certificate_disagrees')
    _require(_equal(report['stats']['candidates_checked'], fresh['stats']['candidates_checked']),
             'collection_coverage_work_count_disagrees')
    budget.check()
    return _answer(True, 'supplied_collection_partition_replayed', scope='supplied_family_list_only')


def _known_classes(report, budget):
    if 'dimension' in report and isinstance(report.get('action'), dict):
        from .equivalence import small_class_representatives
        action = report['action']
        fresh = small_class_representatives(report['q'], report['rows'], report['columns'], report['dimension'],
                                             allow_transpose=action['allow_transpose'],
                                             allow_frobenius=action['allow_frobenius'])
        _require(_equal(report, fresh), 'known_classification_disagrees_with_theorem')
        budget.check()
        return _answer(True if report['status'] == 'COMPLETE_CLASSIFICATION' else None,
                       'small_generator_classification_replayed')
    from .research import known_classes
    q, m, n = report['q'], report['rows'], report['columns']
    k = report.get('size')
    exemplar = report
    if 'sizes' in report:
        _require(isinstance(report['sizes'], list) and report['sizes'])
        exemplar = report['sizes'][0]
    action = exemplar['action']
    _require(isinstance(action, str))
    transpose, frobenius = 'with_transpose' in action, action.startswith('semilinear')
    representatives = exemplar.get('representatives', [])
    spec = representatives[0]['field'] if representatives else None
    fresh = known_classes(q, m, n, k, allow_transpose=transpose,
                          allow_frobenius=frobenius, field_spec=spec)
    _require(_equal(report, fresh), 'known_classification_disagrees_with_theorem')
    if report['status'] == 'INCONCLUSIVE':
        return _answer(None, 'classification_domain_unresolved')
    # The theorem is re-evaluated, rather than trusting any stored class count.
    budget.check()
    return _answer(True, 'known_classification_theorem_replayed', scope=report.get('scope'))


def _parameter_classes(report, budget):
    from .classification import classify_parameter_space
    q, m, n, field = _parameters(report)
    _require(report['scope'] == 'all_good_spaces_at_declared_field_shape_and_dimension')
    complete = report['status'] == 'COMPLETE_CLASSIFICATION'
    _require(type(report['complete']) is bool and report['complete'] == complete)
    _require(type(report['exhausted']) is bool and report['exhausted'] == complete)
    if not complete:
        _require(report['status'] == 'PARTIAL_CLASSIFICATION' and report['class_count'] is None)
        return _answer(None, 'parameter_domain_enumeration_is_incomplete')
    _require(_integer(report['class_count']) and report['class_count'] == len(report['classes']))
    action = report['action']
    fresh = classify_parameter_space(q, m, n, report['size'], field_spec=field,
                                     allow_transpose=action['allow_transpose'],
                                     allow_frobenius=action['allow_frobenius'],
                                     max_candidates=budget.count, max_nodes=budget.count,
                                     max_seconds=budget.remaining())
    if fresh['status'] != 'COMPLETE_CLASSIFICATION':
        return _answer(None, 'complete_classification_replay_limit')
    for key in ('method', 'q', 'field', 'rows', 'columns', 'size', 'action', 'status',
                'scope', 'domain', 'class_count', 'class_count_lower_bound', 'classes', 'coverage'):
        _require(_equal(report[key], fresh[key]), 'complete_classification_or_coverage_disagrees')
    for key, value in fresh['stats'].items():
        if key != 'elapsed_seconds':
            _require(_equal(report['stats'][key], value), 'classification_work_count_disagrees')
    budget.check()
    return _answer(True, 'complete_parameter_domain_classification_replayed')


def _construct(report, budget):
    # Constructor predictions have their own small schema. They must never
    # become a fallback route for a checker certificate with a damaged method.
    _require(not any(key in report for key in ('method', 'schema_version', 'stats', 'software')),
             'checker_evidence_cannot_be_reinterpreted_as_a_constructor')
    if 'range' in report:
        _require(set(report) == {'status', 'reason', 'problem', 'certificate', 'range'} and
                 report['problem'].get('construction', {}).get('family') == 'extended_vandermonde' and
                 report['certificate'].get('proof') == 'exact_rank_one_hankel_range',
                 'unrecognized_constructor_certificate_schema')
    else:
        _require(set(report) == {'status', 'reason', 'problem'} and report['reason'] in (
            'Packaged construction; checked again before display.',
            'Two-row characterization and explicit construction.'),
            'unrecognized_constructor_prediction_schema')
    from .fibre import fibre_check, verify_fibre
    checked = fibre_check(report['problem'], max_fibres=budget.count, max_seconds=budget.remaining())
    if checked['status'] == 'INCONCLUSIVE':
        return _answer(None, 'constructor_replay_limit')
    _require(report['status'] == checked['status'], 'constructor_prediction_disagrees')
    replay = verify_fibre(checked, max_fibres=budget.count)
    if replay['verified'] is not True:
        return replay
    if 'range' in report:
        from .constructions import construct_vandermonde
        semantic = _semantic_problem(report['problem'])
        parameters = report['problem']['construction']['parameters']
        fresh = construct_vandermonde(semantic['field']['order'], *semantic['shape'],
                                      parameters=parameters, field_spec=semantic['field'])
        _require(_same_problem(fresh['problem'], report['problem']))
        for key in ('status', 'reason', 'range', 'certificate'):
            _require(_equal(report[key], fresh[key]), 'constructor_certificate_disagrees')
    budget.check()
    return _answer(True, 'construction_input_and_range_replayed', claim=report['status'])


def _bundle(data, budget):
    operation, report = data['operation'], data['report']
    _require(isinstance(report, dict) and isinstance(data.get('status'), str))
    method = report.get('method', '')
    compatible = {
        'check': method in ('annihilator-fibre', 'binary_ambient_membership', 'hybrid', 'normalized', 'reference-full'),
        'search': method.startswith('projective_quotient_') or method.startswith('binary_partial_matchings_'),
        'bounds': 'witness' in report and 'upper_bound' in report,
        'table': 'entries' in report,
        'equivalence': method == 'paired_projective_transporters',
        'canonical': method == 'paired_projective_canonical',
        'classify': method == 'completed_canonical_keys',
        'enumerate_classes': method == 'complete_projective_prefix_orbit_enumeration',
        'known_classes': 'representatives' in report or 'sizes' in report,
        'distribution': 'matrix_counts' in report or report.get('status') == 'INCONCLUSIVE',
        'factor_analysis': 'left_factor_rank' in report or report.get('status') in ('INCONCLUSIVE', 'INVALID'),
        'construct': 'problem' in report or report.get('status') == 'UNAVAILABLE',
    }
    if operation not in compatible:
        return _answer(None, 'unsupported_saved_bundle_type')
    _require(compatible[operation], 'bundle_operation_and_proof_type_disagree')
    if report.get('status') == 'UNAVAILABLE':
        return _answer(None, 'no_available_result_to_verify')
    if operation == 'construct':
        check = data['check']
        checked_report = check.get('report', check)
        _require(_same_problem(data['problem'], report['problem']) and
                 _same_problem(checked_report['problem'], report['problem']),
                 'construction_bundle_uses_different_inputs')
        _require(data['status'] == check['status'])
        _require(data.get('construction_status', report['status']) == report['status'])
        replay = _dispatch(check, budget)
        if replay['verified'] is not True:
            return replay
        return _construct(report, budget)
    if 'problem' in data and 'problem' in report:
        _require(_same_problem(data['problem'], report['problem']), 'bundle_and_report_inputs_disagree')
    if data['status'] != report.get('status'):
        _require(data['status'] == 'INCONCLUSIVE' and report.get('status') in ('VALID', 'EQUIVALENT'),
                 'bundle_and_report_statuses_disagree')
        return _answer(None, 'bundle_has_no_completed_replay_claim')
    return _dispatch(report, budget)


def _dispatch(data, budget):
    if budget.depth >= 32:
        return _answer(False, 'saved_result_nesting_limit')
    budget.depth += 1
    try:
        return _dispatch_inner(data, budget)
    finally:
        budget.depth -= 1


def _dispatch_inner(data, budget):
    budget.check()
    _require(isinstance(data, dict), 'saved_result_must_be_an_object')
    if 'operation' in data and 'report' in data:
        return _bundle(data, budget)
    method = data.get('method', '')
    if method in ('annihilator-fibre', 'binary_ambient_membership', 'hybrid', 'normalized', 'reference-full'):
        from .verify import verify
        return verify(data, max_directions=budget.count, max_supports=budget.count)
    if method == 'paired_projective_transporters':
        from .equivalence import verify_equivalence
        return verify_equivalence(data, max_candidates=budget.count, max_seconds=budget.remaining())
    if method == 'paired_projective_canonical':
        from .equivalence import verify_canonical
        return verify_canonical(data, max_candidates=budget.count, max_seconds=budget.remaining())
    if method == 'completed_canonical_keys':
        return _collection(data, budget)
    if method == 'complete_projective_prefix_orbit_enumeration':
        return _parameter_classes(data, budget)
    if method in ('projective_quotient_greedy', 'projective_quotient_exhaustive',
                  'binary_partial_matchings_span', 'binary_partial_matchings_quotient'):
        return _bounds(data, budget, search=True)
    if 'problem' in data and any(key in data for key in ('schema_version', 'stats', 'software')):
        return _answer(False, 'unknown_or_missing_certificate_method')
    if 'method' in data and method != 'proved_rank_at_most_two_formula':
        return _answer(None, 'unsupported_certificate_method')
    if 'witness' in data and 'upper_bound' in data:
        return _bounds(data, budget)
    if data.get('status') in ('VALID', 'INVALID') and 'problem' in data:
        return _construct(data, budget)
    if 'representatives' in data or 'sizes' in data and 'q' in data:
        return _known_classes(data, budget)
    if 'entries' in data and data.get('status') == 'COMPLETE':
        if data.get('scope') != 'Numerical bounds for 1 <= rows <= columns <= 5; not an orbit classification.':
            return _answer(None, 'unsupported_saved_table_domain')
        _require(isinstance(data['entries'], list))
        expected = [(m, n) for m in range(1, 6) for n in range(m, 6)]
        _require([(e['rows'], e['columns']) for e in data['entries']] == expected,
                 'numerical_table_coverage_is_incomplete')
        _require(all(e['q'] == data['q'] for e in data['entries']), 'table_fields_disagree')
        for entry in data['entries']:
            replay = _dispatch(entry, budget)
            if replay['verified'] is not True:
                return replay
        budget.check()
        return _answer(True, 'every_saved_table_entry_replayed', entries=len(data['entries']))
    if 'matrix_counts' in data and 'problem' in data and data.get('status') == 'COMPLETE':
        from .checker import rank_distribution
        fresh = rank_distribution(data['problem'], max_directions=budget.count,
                                  method='auto', max_seconds=budget.remaining())
        if fresh['status'] != 'COMPLETE':
            return _answer(None, 'distribution_replay_limit')
        _require(_equal(data['matrix_counts'], fresh['matrix_counts']) and
                 _equal(data['total_matrices'], fresh['total_matrices']), 'rank_distribution_disagrees')
        budget.check()
        return _answer(True, 'exact_rank_distribution_replayed')
    if 'left_factor_rank' in data and 'records' in data and 'problem' in data:
        from .analysis import factor_analysis
        if data.get('status') != 'COMPLETE':
            return _answer(None, 'factor_analysis_incomplete')
        fresh = factor_analysis(data['problem'], max_subsets=budget.count, max_seconds=budget.remaining())
        if fresh['status'] != 'COMPLETE':
            return _answer(None, 'factor_analysis_replay_limit')
        for key, value in fresh.items():
            if key not in ('elapsed_seconds', 'limits'):
                _require(_equal(data[key], value), 'factor_analysis_disagrees')
        budget.check()
        return _answer(True, 'factor_rank_functions_replayed')
    return _answer(None, 'unsupported_saved_result_type')


def _run(data, budget):
    try:
        out = _dispatch(data, budget)
        budget.check()
        return out
    except _Limit:
        return _answer(None, 'wall_time_limit')
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, ZeroDivisionError, RecursionError) as exc:
        return _answer(False, 'malformed_or_inconsistent_saved_result', detail=str(exc))


def verify_saved(data, *, max_candidates=100000, max_seconds=None):
    """Replay a supported saved result; unknown or unfinished proofs stay None."""
    budget = _Budget(max_candidates, max_seconds)
    if max_seconds is None:
        return _run(data, budget)
    if max_seconds == 0:
        return _answer(None, 'wall_time_limit')
    # One worker enforces the caller's deadline even for older replay routines.
    # It invokes _run directly, so recursive dispatch never creates workers.
    try:
        encoded = json.dumps({'data': data, 'max_candidates': max_candidates,
                              'max_seconds': max_seconds}, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as exc:
        return _answer(False, 'saved_result_is_not_finite_json', detail=str(exc))
    if budget.remaining() == 0:
        return _answer(None, 'wall_time_limit')
    process = None
    try:
        process = subprocess.Popen([sys.executable, '-m', 'rankone.certificates', '--worker'],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        stdout, _ = process.communicate(encoded, timeout=budget.remaining())
        if process.returncode:
            return _answer(None, 'replay_worker_failed')
        answer = json.loads(stdout)
        _require(isinstance(answer, dict) and
                 (answer.get('verified') is None or type(answer.get('verified')) is bool))
        if budget.remaining() == 0:
            return _answer(None, 'wall_time_limit')
        return answer
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        return _answer(None, 'wall_time_limit')
    except (OSError, ValueError, TypeError):
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate()
        return _answer(None, 'replay_worker_unavailable')


def _worker():
    try:
        request = json.loads(sys.stdin.read())
        answer = _run(request['data'], _Budget(request['max_candidates'], request['max_seconds']))
    except Exception:
        answer = _answer(None, 'replay_worker_error')
    print(json.dumps(answer, allow_nan=False))


if __name__ == '__main__':
    _worker()
