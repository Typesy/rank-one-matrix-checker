"""Complete tiny-domain orbit enumeration with honest interruption labels.

The domain is all unordered projective rank-one bases of the declared size,
not a supplied sample. Every good family has one increasing-index ordering;
goodness is hereditary, so singleton quotient-fibre extension is complete.
"""

from time import monotonic

import numpy as np

from .constructions import _dimension,_field,projective_vectors
from .equivalence import canonical_form,verify_equivalence
from .search import _budget,_projective_residuals


def classify_parameter_space(q,m,n,k,*,allow_transpose=False,allow_frobenius=False,
                             max_candidates=100000,max_nodes=100000,
                             max_seconds=None,field_spec=None):
    """Exhaust all good size-k families, then deduplicate completed orbit keys.

    max_candidates bounds the ambient allocation AND combined work: every row
    in a global residual census and every canonical basis-image candidate.
    max_nodes bounds safe prefix extensions. A partial report gives a lower
    bound on the number of classes, never an exact count. Work counts are not
    a formal proof kernel; scope and the search coverage argument are explicit.
    """
    _dimension(m,"rows");_dimension(n,"columns")
    _budget(k,"set size");_budget(max_candidates,"max_candidates");_budget(max_nodes,"max_nodes")
    if type(allow_transpose) is not bool or type(allow_frobenius) is not bool:
        raise ValueError("equivalence options must be booleans")
    if allow_transpose and m!=n:
        raise ValueError("transpose equivalence is available only for square shapes")
    if max_seconds is not None and (type(max_seconds) not in (int,float)
                                    or not 0<=max_seconds<float('inf')):
        raise ValueError("max_seconds must be finite and nonnegative")
    started=monotonic()
    f=_field(q,field_spec)
    nm,nn=(q**m-1)//(q-1),(q**n-1)//(q-1)
    total=nm*nn
    classes=[];by_key={};stop=None;ambient_built=False
    stats={"ambient_rank_one_directions":total,"prefix_extensions":0,
           "residual_censuses":0,"census_rows_checked":0,
           "canonical_candidates_checked":0,"combined_candidate_work":0,
           "good_families_encountered":0,"families_canonicalized":0,
           "incomplete_canonical_leaves":0,"size_prunes":0}
    action={"left_right_linear":True,"allow_transpose":allow_transpose,
            "allow_frobenius":allow_frobenius,"generator_order":"unordered",
            "generator_scaling":"nonzero"}

    def expired():
        return max_seconds is not None and monotonic()-started>=max_seconds

    def remaining_time():
        return None if max_seconds is None else max(0,max_seconds-(monotonic()-started))

    def finish(reason,exhausted):
        return {"schema_version":1,"method":"complete_projective_prefix_orbit_enumeration",
                "q":q,"field":f.metadata(),"rows":m,"columns":n,"size":k,
                "action":action,"status":"COMPLETE_CLASSIFICATION" if exhausted else "PARTIAL_CLASSIFICATION",
                "reason":reason,"exhausted":exhausted,"complete":exhausted,
                "scope":"all_good_spaces_at_declared_field_shape_and_dimension",
                "domain":{"unordered_projective_rank_one_families":True,
                          "generator_count":k,"field_model":f.metadata(),
                          "rows":m,"columns":n},
                "class_count":len(classes) if exhausted else None,
                "class_count_lower_bound":len(classes),"classes":classes,
                "coverage":{"ordering":"strictly_increasing_ambient_candidate_indices",
                            "ambient_directions_complete":ambient_built,
                            "extension_test":"nonzero_singleton_projective_quotient_fibre",
                            "all_ambient_rows_retained_at_every_prefix":True,
                            "safe_prunes":"insufficient_remaining_distinct_left_or_right_directions",
                            "canonical_keys_used_only_when_complete":True},
                "stats":{**stats,"elapsed_seconds":monotonic()-started},
                "limits":{"max_candidates":max_candidates,"max_nodes":max_nodes,
                          "max_seconds":max_seconds}}

    def add_leaf(problem):
        nonlocal stop
        if expired():
            stop="time_limit";return
        stats["good_families_encountered"]+=1
        available=max_candidates-stats["combined_candidate_work"]
        out=canonical_form(problem,allow_transpose=allow_transpose,
                           allow_frobenius=allow_frobenius,
                           max_candidates=max(0,available),max_seconds=remaining_time())
        used=out['stats']['candidates_checked']
        stats['canonical_candidates_checked']+=used
        stats['combined_candidate_work']+=used
        if out['status']!='COMPLETE':
            if out['status']=='INVALID_INPUT':
                raise ArithmeticError("canonical validation rejected a good prefix leaf")
            stats['incomplete_canonical_leaves']+=1
            stop="time_limit" if expired() else "canonical_work_limit"
            return
        # A positive transporter can be independently replayed even though
        # proving a canonical minimum requires the complete search above.
        replay=verify_equivalence({"schema_version":1,"status":"EQUIVALENT",
                                  "source":out['problem'],"target":out['representative'],
                                  "action":out['action'],"certificate":out['transporter']})
        if replay['verified'] is not True:
            raise ArithmeticError("canonical transporter replay failed")
        stats['families_canonicalized']+=1
        key=out['canonical_key']
        if key not in by_key:
            by_key[key]=len(classes)
            classes.append({"class_index":len(classes),"canonical_key":key,
                            "representative":out['representative'],
                            "first_family":problem,"transporter":out['transporter'],
                            "transporter_replay":replay,"families_in_class":0})
        classes[by_key[key]]['families_in_class']+=1

    if expired():
        return finish("time_limit",False)
    if k==0:
        add_leaf({"field":f.metadata(),"rows":m,"columns":n,"generators":[]})
        return finish(stop or "zero_space_domain_exhausted",stop is None)
    # Distinct factors and independent flattened generators are necessary.
    # A root size prune is a complete exclusion of this parameter domain.
    if k>min(m*n,nm,nn):
        stats['size_prunes']+=1
        return finish("required_size_exceeds_distinct_factor_or_dimension_bound",True)
    if total>max_candidates:
        return finish("ambient_candidate_limit",False)
    if max_nodes==0:
        return finish("node_limit",False)
    z=f.GF.Range(0,q)
    mul=np.asarray(z[:,None]*z[None,:],dtype=np.int64)
    sub=np.asarray(z[:,None]-z[None,:],dtype=np.int64)
    inverses=np.array([0]+[f.inv(x) for x in range(1,q)],dtype=np.int64)
    left=np.array(list(projective_vectors(q,m)),dtype=np.int64)
    right=np.array(list(projective_vectors(q,n)),dtype=np.int64)
    ambient=mul[left[:,None,:,None],right[None,:,None,:]].reshape(total,m*n)
    ambient_built=True

    def visit(chosen,residual,last,used_left,used_right):
        nonlocal stop
        if expired():
            stop='time_limit';return
        if len(chosen)==k:
            add_leaf({"field":f.metadata(),"rows":m,"columns":n,
                      "generators":ambient[chosen].reshape(k,m,n).tolist()})
            return
        available=min(nm-len(used_left),nn-len(used_right),total-last-1)
        if len(chosen)+available<k:
            stats['size_prunes']+=1;return
        if stats['combined_candidate_work']+total>max_candidates:
            stop='combined_candidate_limit';return
        stats['residual_censuses']+=1
        stats['census_rows_checked']+=total
        stats['combined_candidate_work']+=total
        eligible,normalized,pivots=_projective_residuals(residual,mul,inverses)
        for value in eligible:
            index=int(value)
            if index<=last:
                continue
            li,ri=divmod(index,nn)
            if li in used_left or ri in used_right:
                continue
            if expired():
                stop='time_limit';return
            if stats['prefix_extensions']>=max_nodes:
                stop='node_limit';return
            stats['prefix_extensions']+=1
            pivot=pivots[index]
            extended=sub[residual,mul[residual[:,pivot][:,None],normalized[index][None,:]]]
            visit(chosen+[index],extended,index,used_left|{li},used_right|{ri})
            if stop:
                return
    visit([],ambient,-1,set(),set())
    return finish(stop or 'all_good_families_and_canonical_keys_exhausted',stop is None)
