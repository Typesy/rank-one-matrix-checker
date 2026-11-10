"""Independent mathematical oracles for constructors, bounds and search."""

from copy import deepcopy
from itertools import combinations, product

import pytest

from rankone.constructions import (construct_two_row,construct_vandermonde,
                                   projective_vectors,two_row_classes)
from rankone.field import Field
from rankone.fibre import fibre_check,verify_fibre
from rankone.linalg import combine,flatten,rank
from rankone.research import known_classes,known_results,load_fixture,load_witness,upper_bound
from rankone.search import _projective_residuals,maximum_search
from rankone.analysis import factor_analysis,two_row_distribution


def coefficient_good(problem):
    """Every coefficient vector, using rank rather than fibre membership."""
    f=Field(problem['field']['order'],problem['field'].get('irreducible_poly'))
    a=problem['generators'];k=len(a)
    if not a:
        return True
    if rank([flatten(x) for x in a],f)!=k or any(rank(x,f)!=1 for x in a):
        return False
    for c in product(range(f.q),repeat=k):
        if sum(x!=0 for x in c)>1 and rank(combine(a,c,f),f)<2:
            return False
    return True


@pytest.mark.parametrize('q',[2,3,4,5])
def test_two_row_all_small_sizes_and_classes(q):
    for n in range(1,7):
        maximum=n if n<q else q+1
        assert upper_bound(q,2,n)['upper_bound']==maximum
        for k in range(maximum+1):
            result=two_row_classes(q,n,k)
            expected=2 if k==q+1 and n>=q+1 else 1
            assert result['status']=='COMPLETE_CLASSIFICATION'
            assert result['class_count']==expected
            for a in result['representatives']:
                report=fibre_check(a)
                assert report['status']=='VALID'
                assert verify_fibre(report)['verified'] is True
                if q**k<=5000:
                    assert coefficient_good(a)
        assert not two_row_classes(q,n,maximum+1)['representatives']


@pytest.mark.parametrize('q',[2,3,4,5])
def test_vandermonde_boundaries_and_explicit_certificates(q):
    for m,n in [(2,2),(2,3),(3,3),(3,4),(5,5)]:
        d=m+n-1
        for k in sorted({0,1,min(q+1,d-1),min(q+1,d),min(q+1,d+1)}):
            out=construct_vandermonde(q,m,n,k=k,include_infinity=True)
            expected=k<=d-1 or k==q+1==d
            assert (out['status']=='VALID')==expected
            checked=fibre_check(out['problem'])
            assert checked['status']==out['status']
            assert verify_fibre(checked)['verified'] is True
            if not expected:
                c=out['certificate']['coefficients']
                f=Field(q)
                actual=combine(out['problem']['generators'],c,f)
                if out['reason']=='dependent_generators':
                    assert not any(flatten(actual)) and any(c)
                else:
                    assert actual==out['certificate']['matrix']
                    assert rank(actual,f)==1 and sum(bool(x) for x in c)>1


def test_vandermonde_distinctness_and_range_are_field_aware():
    with pytest.raises(ValueError):construct_vandermonde(4,2,2,[1,1])
    with pytest.raises(ValueError):construct_vandermonde(4,2,2,[None,'infinity'])
    with pytest.raises(ValueError):construct_vandermonde(4,1,3)
    with pytest.raises(ValueError):construct_vandermonde(4,2,3,[0],k=1)
    with pytest.raises(ValueError):construct_vandermonde(4,2,3,[4])
    assert construct_vandermonde(4,3,3,include_infinity=True)['status']=='VALID'
    assert construct_vandermonde(5,3,3,include_infinity=True)['status']=='INVALID'
    # All finite parameters may be good at large degree: no d<=q restriction.
    assert construct_vandermonde(3,5,5)['status']=='VALID'


@pytest.mark.parametrize('q',[2,3,4,5])
def test_packaged_witnesses_and_proved_bounds(q):
    expected=([7,7,7,12,15,20] if q==2 else [6,8,10,11,14,18])
    shapes=[(3,3),(3,4),(3,5),(4,4),(4,5),(5,5)]
    for (m,n),lower in zip(shapes,expected):
        result=known_results(q,m,n)
        assert result['lower_bound']==lower
        assert result['upper_bound']==([7,7,7,12,15,20] if q==2 else [6,8,10,12,15,20])[shapes.index((m,n))]
        checked=fibre_check(result['witness'])
        assert checked['status']=='VALID'
        assert verify_fibre(checked)['verified'] is True
        reverse=load_witness(q,n,m)
        assert len(reverse['generators'])==lower
        assert len(reverse['generators'][0])==n


def test_bounds_outside_ledger_and_field_models():
    assert known_results(2,3,9)['status']=='EXACT'
    assert known_results(2,3,9)['lower_bound']==7
    assert known_results(7,2,7)['lower_bound']==8
    assert known_results(4,1,12)['lower_bound']==1
    for q in [6,10,12]:
        with pytest.raises(ValueError):known_results(q,2,2)
    with pytest.raises(ValueError):construct_two_row(4,3,field_spec={'order':3})
    with pytest.raises(ValueError):construct_two_row(4,3,field_spec={'order':4,'characteristic':4})


def test_small_complete_classification_counts_and_unknown_scope():
    assert known_classes(2,3,3,3)['class_count']==4
    assert known_classes(3,3,3,3)['class_count']==3
    assert known_classes(2,3,3,3,allow_transpose=True)['class_count']==3
    assert known_classes(3,3,3,3,allow_transpose=True)['class_count']==2
    assert known_classes(5,3,3,6)['status']=='INCONCLUSIVE'
    assert known_classes(5,3,3,7)['class_count']==0
    assert known_classes(2,2,3,3)['class_count']==2
    assert known_classes(3,2,3,4)['class_count']==1
    assert known_classes(4,2,3,4)['class_count']==0
    assert known_classes(2,3,3)['status']=='PARTIAL_CLASSIFICATION'
    assert known_classes(2,2,3)['status']=='COMPLETE_CLASSIFICATION'
    with pytest.raises(ValueError):known_classes(2,2,3,2,allow_transpose=True)


def test_named_mds_fixtures_and_no_path_injection():
    for name in ['f5_mds_identity','f5_mds_transposition']:
        assert fibre_check(load_fixture(name))['status']=='VALID'
    with pytest.raises(ValueError):load_fixture('../field')


def brute_force_maximum_2x2(q):
    """Whole ambient sets + whole coefficients, no quotient search."""
    f=Field(q);vectors=list(projective_vectors(q,2))
    matrices=[[[f.mul(x,y) for y in v] for x in u] for u in vectors for v in vectors]
    largest=0
    # The elementary distinct-factor bound gives k<=q+1. We only need size3
    # to distinguish binary exception from nonbinary two-row theorem.
    for k in range(1,min(3,q+1)+1):
        for indices in combinations(range(len(matrices)),k):
            if coefficient_good({'field':{'order':q},'generators':[matrices[i] for i in indices]}):
                largest=k;break
    return largest


@pytest.mark.parametrize('q',[2,3,4])
def test_exhaustive_search_agrees_with_independent_tiny_oracle(q):
    result=maximum_search(q,2,2,method='exhaustive',use_known=False,max_nodes=10000)
    assert result['status']=='EXACT'
    assert result['lower_bound']==brute_force_maximum_2x2(q)
    assert result['witness_replay']['verified'] is True


def test_general_residuals_identify_proportional_not_just_equal_vectors():
    import numpy as np
    f=Field(5);z=f.GF.Range(0,5)
    mul=np.asarray(z[:,None]*z[None,:],dtype=np.int64)
    inverses=np.array([0]+[f.inv(i) for i in range(1,5)])
    residual=np.array([[1,2],[2,4],[0,1],[0,0]],dtype=np.int64)
    eligible,normalized,_=_projective_residuals(residual,mul,inverses)
    assert eligible.tolist()==[2]
    assert normalized[0].tolist()==normalized[1].tolist()


def test_search_limits_and_known_result_fast_path():
    exact=maximum_search(4,3,3,max_candidates=0)
    assert exact['status']=='EXACT' and exact['lower_bound']==6
    for options in [{'max_seconds':0},{'max_nodes':0},{'max_candidates':0}]:
        out=maximum_search(3,2,2,use_known=False,**options)
        assert out['status']=='INCONCLUSIVE' and not out['maximum_proved']
    partial=maximum_search(3,4,4,max_candidates=0)
    assert partial['status']=='INCONCLUSIVE'
    assert (partial['lower_bound'],partial['upper_bound'])==(11,12)
    out=maximum_search(3,2,2,use_known=False,trials=0)
    assert out['status']=='BOUNDED' and not out['maximum_proved']
    prefix=maximum_search(3,3,3,use_known=False,max_nodes=1)
    assert prefix['status']=='INCONCLUSIVE' and prefix['lower_bound']==1
    assert prefix['witness_replay']['verified'] is True
    for bad in [-1,True,1.2]:
        with pytest.raises(ValueError):maximum_search(3,2,2,max_nodes=bad)


def test_greedy_is_reproducible_and_never_claims_a_failed_search_exact():
    a=maximum_search(3,3,3,use_known=False,trials=2,seed=12,max_nodes=100)
    b=maximum_search(3,3,3,use_known=False,trials=2,seed=12,max_nodes=100)
    assert a['lower_bound']==b['lower_bound']
    assert a['witness']==b['witness']
    assert a['status'] in ['EXACT','BOUNDED']
    if a['status']=='EXACT':assert a['lower_bound']==a['upper_bound']==6
    assert a['witness_replay']['verified'] is True


def test_factor_matroid_bounds_are_sufficient_not_necessary():
    binary=construct_two_row(2,2)
    info=factor_analysis(binary)
    assert info['status']=='COMPLETE'
    assert not info['sufficient_condition_proved']
    assert fibre_check(binary)['status']=='VALID'
    independent=construct_two_row(3,2)
    info=factor_analysis(independent)
    assert info['sufficient_condition_proved']
    assert info['left_factor_rank']==info['right_factor_rank']==2
    assert info['subsets_checked']==4
    partial=factor_analysis(binary,max_subsets=2)
    assert partial['status']=='INCONCLUSIVE' and not partial['sufficient_condition_proved']
    timed=factor_analysis(binary,max_seconds=0)
    assert timed['status']=='INCONCLUSIVE'


def test_matroid_profiles_do_not_claim_equivalence():
    first=factor_analysis(load_fixture('f5_mds_identity'))
    second=factor_analysis(load_fixture('f5_mds_transposition'))
    assert first['records']==second['records']
    assert first['equivalence_conclusion']=='NOT_DETERMINED_BY_MATROID_DATA'


@pytest.mark.parametrize('q',[2,3,4,5])
def test_two_row_distribution_formula_and_padded_support(q):
    problem=construct_two_row(q,q+1)
    out=two_row_distribution(problem)
    assert out['status']=='COMPLETE'
    assert out['matrix_counts']['1']==(q+1)*(q-1)
    assert sum(out['matrix_counts'].values())==q**(q+1)
    padded=deepcopy(problem)
    padded['rows']=3
    for matrix in padded['generators']:matrix.append([0]*(q+1))
    assert two_row_distribution(padded)['matrix_counts']==out['matrix_counts']


def test_distribution_checks_goodness_and_scope_first():
    invalid={'field':{'order':3},'generators':[[[1,0],[0,0]],[[0,0],[0,1]],[[1,1],[1,1]]]}
    assert two_row_distribution(invalid)['status']=='INVALID'
    with pytest.raises(ValueError):two_row_distribution(load_witness(3,3,3))
    assert two_row_distribution(construct_two_row(2,2),max_fibres=0)['status']=='INCONCLUSIVE'
    zero={'field':{'order':4},'rows':2,'columns':3,'generators':[]}
    assert two_row_distribution(zero)['matrix_counts']=={'0':1}
