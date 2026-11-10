"""Bounded general-field search using complete projective quotient fibres.

The global ambient census is retained at every prefix. A nonzero singleton
projective residual fibre is exactly a safe rank-one extension. Greedy failure
is only a lower bound; completed exhaustive search supplies an upper bound.
"""

from time import monotonic

import numpy as np

from .constructions import _dimension, _field, projective_vectors
from .research import known_results


def _budget(value,name):
    if type(value) is not int or value<0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _projective_residuals(residual,multiply,inverses):
    nonzero=np.any(residual,axis=1)
    pivots=np.argmax(residual!=0,axis=1)
    first=residual[np.arange(len(residual)),pivots]
    normalized=multiply[residual,inverses[first][:,None]]
    _,classes,counts=np.unique(normalized,axis=0,return_inverse=True,return_counts=True)
    eligible=np.flatnonzero(nonzero & (counts[classes]==1))
    return eligible,normalized,pivots


def maximum_search(q,m,n,*,method="greedy",trials=20,seed=20261026,
                   max_nodes=100000,max_candidates=100000,max_seconds=None,
                   use_known=True,field_spec=None):
    """Find checked constructions or finish a bounded exact search.

    Known constructions are checked and replayed before supplying a lower
    bound. Exhaustive mode explores increasing ambient candidate indices;
    every good set has exactly one such ordering. Limits are INCONCLUSIVE.
    max_nodes counts prefix attempts (greedy: tested singleton choices).
    A successful greedy run that has not attained a proof bound is BOUNDED.
    """
    _dimension(m,"rows");_dimension(n,"columns")
    if method not in ("greedy","exhaustive"):
        raise ValueError("method must be greedy or exhaustive")
    for name,value in (("trials",trials),("seed",seed),("max_nodes",max_nodes),
                       ("max_candidates",max_candidates)):
        _budget(value,name)
    if max_seconds is not None and (type(max_seconds) not in (int,float)
                                    or not 0<=max_seconds<float("inf")):
        raise ValueError("max_seconds must be finite and nonnegative")
    if type(use_known) is not bool:
        raise ValueError("use_known must be a boolean")
    f=_field(q,field_spec)
    knowledge=known_results(q,m,n,field_spec=f.metadata())
    upper=knowledge["upper_bound"]
    nm,nn=(q**m-1)//(q-1),(q**n-1)//(q-1)
    total=nm*nn
    started=monotonic()
    limits={"max_nodes":max_nodes,"max_candidates":max_candidates,
            "max_seconds":max_seconds,"trials":trials}
    stats={"candidate_count":total,"extensions_tested":0,"trials_completed":0,
           "valid_prefixes":0,"bound_prunes":0}
    best_problem={"field":f.metadata(),"rows":m,"columns":n,"generators":[]}
    best_check=None;best_replay=None

    def expired():
        return max_seconds is not None and monotonic()-started>=max_seconds

    def verify_problem(problem):
        from .fibre import fibre_check,verify_fibre
        if expired():
            return None,None
        remaining=None if max_seconds is None else max(0,max_seconds-(monotonic()-started))
        checked=fibre_check(problem,max_fibres=100000,max_seconds=remaining)
        if checked["status"]=="INVALID":
            raise ArithmeticError("independent checking rejected a search witness")
        if checked["status"]!="VALID" or expired():
            return checked,None
        replay=verify_fibre(checked,max_fibres=100000)
        if replay["verified"] is False:
            raise ArithmeticError("independent replay rejected a search witness")
        return checked,replay

    def finish(reason,*,exhausted=False):
        lower=len(best_problem["generators"])
        verified=(best_check is not None and best_check["status"]=="VALID" and
                  best_replay is not None and best_replay["verified"] is True)
        # The empty basis is an exact mathematical lower bound zero even if
        # an expired computation has not built a software certificate for it.
        if lower and not verified:
            raise ArithmeticError("an unchecked positive lower bound cannot be reported")
        exact=(lower==upper and verified) or exhausted
        limited=reason in ("time_limit","node_limit","candidate_limit","witness_verification_limit")
        return {"schema_version":1,"method":"projective_quotient_"+method,
                "field":f.metadata(),"q":q,"rows":m,"columns":n,
                "status":"EXACT" if exact else "INCONCLUSIVE" if limited else "BOUNDED",
                "reason":reason,"lower_bound":lower,"upper_bound":lower if exhausted else upper,
                "upper_bound_proof":({"proof":"complete_increasing_prefix_search",
                                     "complete":True} if exhausted else knowledge["upper_bound_proof"]),
                "witness":best_problem,"witness_check":best_check,"witness_replay":best_replay,
                "search_complete":exhausted,"maximum_proved":exact,
                "classification_status":"NOT_CLASSIFIED",
                "stats":{**stats,"elapsed_seconds":monotonic()-started},"limits":limits,
                "seed":seed,"uses_stored_constructions":use_known}

    if expired():
        return finish("time_limit")
    if use_known:
        problem=knowledge["witness"]
        checked,replay=verify_problem(problem)
        if checked is None or checked["status"]!="VALID" or replay is None or replay["verified"] is not True:
            return finish("witness_verification_limit")
        best_problem,best_check,best_replay=problem,checked,replay
        if len(problem["generators"])==upper:
            return finish("proved_upper_bound_attained_by_checked_construction")
    if total>max_candidates:
        return finish("candidate_limit")
    if not max_nodes:
        return finish("node_limit")
    if expired():
        return finish("time_limit")
    # Form every outer-product direction once, normalized factors on both
    # sides. All rows remain in every residual census, even after rejection.
    z=f.GF.Range(0,q)
    multiply=np.asarray(z[:,None]*z[None,:],dtype=np.int64)
    subtract=np.asarray(z[:,None]-z[None,:],dtype=np.int64)
    inverses=np.array([0]+[f.inv(a) for a in range(1,q)],dtype=np.int64)
    left=np.array(list(projective_vectors(q,m)),dtype=np.int64)
    right=np.array(list(projective_vectors(q,n)),dtype=np.int64)
    ambient=multiply[left[:,None,:,None],right[None,:,None,:]].reshape(total,m*n)
    rng=np.random.default_rng(seed)
    stop=None

    def retain(chosen):
        nonlocal best_problem,best_check,best_replay,stop
        if len(chosen)<=len(best_problem["generators"]):
            return
        problem={"field":f.metadata(),"rows":m,"columns":n,
                 "generators":ambient[chosen].reshape(-1,m,n).tolist()}
        checked,replay=verify_problem(problem)
        if checked is None or checked["status"]!="VALID" or replay is None or replay["verified"] is not True:
            stop="witness_verification_limit"
            return
        best_problem,best_check,best_replay=problem,checked,replay
        if len(chosen)==upper:
            stop="proved_upper_bound_attained"

    def extend(residual,index,normalized,pivots):
        pivot=pivots[index]
        return subtract[residual,multiply[residual[:,pivot][:,None],normalized[index][None,:]]]

    if method=="greedy":
        for _ in range(trials):
            if expired():
                stop="time_limit";break
            residual=ambient.copy();chosen=[]
            while True:
                if expired():
                    stop="time_limit";break
                if stats["extensions_tested"]>=max_nodes:
                    stop="node_limit";break
                eligible,normalized,pivots=_projective_residuals(residual,multiply,inverses)
                if not len(eligible):
                    retain(chosen);break
                index=int(rng.choice(eligible))
                stats["extensions_tested"]+=1
                stats["valid_prefixes"]+=1
                chosen.append(index)
                residual=extend(residual,index,normalized,pivots)
                if len(chosen)>=upper:
                    retain(chosen);break
            if chosen and stop=="node_limit":
                # Preserve a useful checked prefix when the node budget ends
                # before that greedy run reaches a maximal family.
                retain(chosen)
            stats["trials_completed"]+=1
            if stop:
                break
        return finish(stop or "greedy_trials_completed")

    def visit(chosen,residual,last,used_left,used_right):
        nonlocal stop
        if expired():
            stop="time_limit";return
        retain(chosen)
        if stop:
            return
        available=min(nm-len(used_left),nn-len(used_right))
        if len(chosen)+available<=len(best_problem["generators"]):
            stats["bound_prunes"]+=1;return
        eligible,normalized,pivots=_projective_residuals(residual,multiply,inverses)
        for value in eligible:
            index=int(value)
            if index<=last:
                continue
            li,ri=divmod(index,nn)
            if li in used_left or ri in used_right:
                continue
            if expired():
                stop="time_limit";return
            if stats["extensions_tested"]>=max_nodes:
                stop="node_limit";return
            stats["extensions_tested"]+=1;stats["valid_prefixes"]+=1
            visit(chosen+[index],extend(residual,index,normalized,pivots),index,
                  used_left|{li},used_right|{ri})
            if stop:
                return
    visit([],ambient,-1,set(),set())
    return finish(stop or "search_exhausted",exhausted=stop is None)
