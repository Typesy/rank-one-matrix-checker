"""Bounded factor-matroid information and proved low-rank distributions."""

from itertools import combinations
from time import monotonic

from .linalg import factors,flatten,rank,rank_one
from .problem import prepare


def _limits(max_subsets,max_seconds):
    if type(max_subsets) is not int or max_subsets<0:
        raise ValueError("max_subsets must be a nonnegative integer")
    if max_seconds is not None and (type(max_seconds) not in (int,float)
                                    or not 0<=max_seconds<float('inf')):
        raise ValueError("max_seconds must be finite and nonnegative")


def factor_analysis(problem,*,max_subsets=100000,max_seconds=None):
    """Record both represented matroid rank functions with their common labels.

    A support rank-sum at least |S|+2 certifies every mixed combination on S.
    Failed bounds remain unresolved; they do not establish invalidity. Even a
    complete pair of abstract rank functions is not an equivalence certificate.
    """
    _limits(max_subsets,max_seconds)
    started=monotonic()
    f,a,canonical=prepare(problem)
    k=len(a)
    records=[];safe=unresolved=0;us=[];vs=[]
    for i,matrix in enumerate(a):
        if not rank_one(matrix,f):
            return {"status":"INVALID","reason":"generator_not_rank_one","index":i,
                    "problem":canonical}
        u,v=factors(matrix,f);us.append(u);vs.append(v)
    flat_rank=rank([flatten(matrix) for matrix in a],f)
    ranks=(rank(us,f),rank(vs,f))
    total=2**k

    def finish(complete,reason):
        condition=complete and unresolved==0
        return {"status":"COMPLETE" if complete else "INCONCLUSIVE","reason":reason,
                "problem":canonical,"generator_count":k,"dimension":flat_rank,
                "independent":flat_rank==k,"left_factor_rank":ranks[0],
                "right_factor_rank":ranks[1],"left_factors":us,"right_factors":vs,
                "records":records,"subsets_checked":len(records),"subsets_total":total,
                "mixed_supports_certified":safe,"mixed_supports_unresolved":unresolved,
                "sufficient_condition_proved":condition,
                "goodness_conclusion":"VALID_BY_SUPPORT_RANK_BOUND" if condition else "NOT_DETERMINED_BY_THIS_SUFFICIENT_TEST",
                "equivalence_conclusion":"NOT_DETERMINED_BY_MATROID_DATA",
                "elapsed_seconds":monotonic()-started,
                "limits":{"max_subsets":max_subsets,"max_seconds":max_seconds}}

    for size in range(k+1):
        for subset in combinations(range(k),size):
            if len(records)>=max_subsets:
                return finish(False,"subset_limit")
            if max_seconds is not None and monotonic()-started>=max_seconds:
                return finish(False,"time_limit")
            ru=rank([us[i] for i in subset],f)
            rv=rank([vs[i] for i in subset],f)
            lower=ru+rv-size
            certified=size>=2 and lower>=2
            if size>=2:
                if certified:safe+=1
                else:unresolved+=1
            records.append({"support":list(subset),"rank_u":ru,"rank_v":rv,
                            "matrix_rank_lower_bound":max(0,lower),
                            "mixed_support_certified":certified})
    return finish(True,"all_factor_rank_functions_computed")


def two_row_distribution(problem,*,max_fibres=100000,max_seconds=None):
    """Exact formula if goodness is certified and all members have rank <=2.

    The condition is tested from the span of the factors, so zero-padding a
    two-row construction does not unnecessarily force coefficient enumeration.
    """
    from .fibre import fibre_check,verify_fibre
    _limits(max_fibres,max_seconds)
    f,a,canonical=prepare(problem)
    checked=fibre_check(canonical,max_fibres=max_fibres,max_seconds=max_seconds)
    if checked['status']!='VALID':
        return {"status":"INCONCLUSIVE" if checked['status']=='INCONCLUSIVE' else "INVALID",
                "reason":"good_basis_required_for_formula","problem":canonical,
                "goodness_check":checked}
    replay=verify_fibre(checked,max_fibres=max_fibres)
    if replay['verified'] is not True:
        return {"status":"INCONCLUSIVE","reason":"goodness_replay_incomplete",
                "problem":canonical,"goodness_check":checked,"replay":replay}
    pairs=[factors(matrix,f) for matrix in a]
    ru=rank([u for u,v in pairs],f)
    rv=rank([v for u,v in pairs],f)
    if min(ru,rv)>2:
        raise ValueError("the analytical distribution formula requires a factor span of dimension at most two")
    k=len(a);one=k*(f.q-1);two=f.q**k-1-one
    if two<0:
        raise ArithmeticError("rank distribution formula contradicts a certified basis")
    counts={"0":1}
    if one:counts["1"]=one
    if two:counts["2"]=two
    return {"status":"COMPLETE","method":"proved_rank_at_most_two_formula",
            "problem":canonical,"matrix_counts":counts,"total_matrices":f.q**k,
            "factor_span_ranks":[ru,rv],"goodness_check":checked,"replay":replay,
            "proof":"Goodness gives k(q-1) rank-one matrices; every other nonzero member has rank two because a factor span has dimension at most two."}
