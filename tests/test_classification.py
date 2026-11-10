"""Independent tiny-domain coverage and partial-classification checks."""

from itertools import combinations,product

import pytest

from rankone.classification import classify_parameter_space
from rankone.constructions import projective_vectors
from rankone.equivalence import verify_equivalence
from rankone.field import Field
from rankone.linalg import combine,flatten,rank


def _all_good_families(q,m,n,k):
    """Independent all-subsets/all-coefficients oracle, no residual quotient."""
    f=Field(q)
    matrices=[[[f.mul(a,b) for b in v] for a in u]
              for u in projective_vectors(q,m) for v in projective_vectors(q,n)]
    good=[]
    for indices in combinations(range(len(matrices)),k):
        family=[matrices[i] for i in indices]
        if rank([flatten(a) for a in family],f)!=k:
            continue
        for coefficients in product(range(q),repeat=k):
            if sum(bool(x) for x in coefficients)>1 and rank(combine(family,coefficients,f),f)<2:
                break
        else:
            good.append(family)
    return good


@pytest.mark.parametrize('q,m,n,k,classes,families',[
    (2,2,2,1,1,9),(2,2,2,2,1,18),(2,2,2,3,1,6),
    (3,2,2,2,1,72),(2,2,3,3,2,210),
])
def test_complete_domain_counts_against_independent_coefficient_oracle(q,m,n,k,classes,families):
    out=classify_parameter_space(q,m,n,k)
    expected=_all_good_families(q,m,n,k)
    assert len(expected)==families
    assert out['status']=='COMPLETE_CLASSIFICATION' and out['exhausted']
    assert out['class_count']==classes
    assert out['class_count_lower_bound']==classes
    assert out['stats']['good_families_encountered']==families
    assert out['stats']['families_canonicalized']==families
    assert sum(c['families_in_class'] for c in out['classes'])==families
    assert out['coverage']['ambient_directions_complete']
    assert out['coverage']['all_ambient_rows_retained_at_every_prefix']
    for c in out['classes']:
        replay=verify_equivalence({'schema_version':1,'status':'EQUIVALENT',
                                  'source':c['first_family'],'target':c['representative'],
                                  'action':out['action'],'certificate':c['transporter']})
        assert replay['verified'] is True
    if (q,m,n,k)==(2,2,3,3):
        # The complete coefficient oracle has two intrinsic right-support
        # ranks. They prove inequivalence and account for the two classes.
        right_ranks=set()
        from rankone.linalg import factors
        for family in expected:
            right_ranks.add(rank([factors(a,Field(q))[1] for a in family],Field(q)))
        assert right_ranks=={2,3}


def test_partial_counts_are_lower_bounds_and_incomplete_keys_are_not_classes():
    for options in [{'max_nodes':0},{'max_seconds':0},{'max_candidates':0},
                    {'max_nodes':3},{'max_candidates':30}]:
        out=classify_parameter_space(2,2,3,3,**options)
        assert out['status']=='PARTIAL_CLASSIFICATION' and not out['exhausted']
        assert out['class_count'] is None
        assert out['class_count_lower_bound']==len(out['classes'])
        assert out['stats']['combined_candidate_work']<=out['limits']['max_candidates']
    # Root census consumes9; no canonical work remains at the first leaf.
    stopped=classify_parameter_space(2,2,2,1,max_candidates=9)
    assert stopped['stats']['good_families_encountered']==1
    assert stopped['stats']['incomplete_canonical_leaves']==1
    assert stopped['classes']==[]
    assert stopped['reason']=='canonical_work_limit'


def test_zero_impossible_and_action_domains():
    zero=classify_parameter_space(4,2,3,0,max_candidates=0)
    assert zero['status']=='COMPLETE_CLASSIFICATION' and zero['class_count']==1
    assert not zero['coverage']['ambient_directions_complete']
    impossible=classify_parameter_space(2,2,2,4)
    assert impossible['status']=='COMPLETE_CLASSIFICATION' and impossible['class_count']==0
    transpose=classify_parameter_space(2,2,2,3,allow_transpose=True)
    assert transpose['class_count']==1
    semilinear=classify_parameter_space(4,1,1,1,allow_frobenius=True)
    assert semilinear['class_count']==1 and semilinear['action']['allow_frobenius']
    with pytest.raises(ValueError):classify_parameter_space(2,2,3,3,allow_transpose=True)
    with pytest.raises(ValueError):classify_parameter_space(6,2,2,2)
    with pytest.raises(ValueError):classify_parameter_space(2,2,2,-1)
    with pytest.raises(ValueError):classify_parameter_space(2,2,2,2,max_candidates=True)


def test_interrupted_search_keeps_completed_classes_without_claiming_completion():
    out=classify_parameter_space(2,2,3,3,max_nodes=10)
    assert out['status']=='PARTIAL_CLASSIFICATION'
    assert out['class_count_lower_bound']>=1
    assert out['class_count'] is None
    assert all(c['canonical_key'] for c in out['classes'])
