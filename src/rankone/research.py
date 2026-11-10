"""Proved bounds, packaged witnesses and complete small-family classifications.

Numerical knowledge is kept separate from search and from equivalence of two
user inputs. Every exact maximum has a construction and a matching bound.
"""

import copy
import json
from importlib.resources import files

from .constructions import _dimension, _field, construct_two_row, two_row_classes
from .linalg import factors


def _data(name):
    return json.loads(files("rankone").joinpath("data",name).read_text())


def _transpose(problem):
    out=copy.deepcopy(problem)
    out["generators"]=[list(map(list,zip(*a))) for a in problem["generators"]]
    if "rows" in out and "columns" in out:
        out["rows"],out["columns"]=out["columns"],out["rows"]
    return out


def upper_bound(q,m,n,*,field_spec=None):
    """A mathematical upper bound with replayable elementary inequalities."""
    _dimension(m,"rows");_dimension(n,"columns")
    f=_field(q,field_spec)
    if min(m,n)==1:
        return {"upper_bound":1,"proof":"one_factor_has_only_one_direction",
                "certificate":{"rows":m,"columns":n,"field":f.metadata()}}
    if min(m,n)==2:
        other=max(m,n)
        value=other if other<q else q+1
        return {"upper_bound":value,"proof":"complete_two_row_maximum_theorem",
                "certificate":{"other_dimension":other,"q":q}}
    count=lambda d:(q**d-1)//(q-1)
    nm,nn=count(m),count(n)
    initial=min(m*n,nm,nn)
    # Goodness is hereditary: the FIRST impossible size excludes that size
    # and every larger size, even if a weaker necessary inequality later holds.
    exclusions=[]
    upper=initial
    for k in range(1,initial+1):
        r=m*n-k
        left=nm*count(max(0,n-r))
        right=nn*count(max(0,m-r))
        if max(left,right)>k:
            upper=k-1
            exclusions=[{"excluded_size":k,"codimension":r,
                         "forced_directions_left":left,
                         "forced_directions_right":right,
                         "good_subfamily_argument":True}]
            break
    return {"upper_bound":upper,"proof":"distinct_factors_and_annihilator_counting",
            "certificate":{"initial_distinct_factor_bound":initial,
                           "left_directions":nm,"right_directions":nn,
                           "exclusions":exclusions}}


def load_fixture(name):
    """Load a named packaged test/demo fixture, without filesystem traversal."""
    if name not in ("f5_mds_identity","f5_mds_transposition","f4_linear_pair_a","f4_linear_pair_b"):
        raise ValueError("unknown named fixture")
    return _data(name+".json")


def load_witness(q,m,n,*,field_spec=None):
    """Return the strongest available construction, zero-padded when useful.

    This function loads or constructs examples; consumers independently check
    them before asserting a computational lower bound from an edited input.
    """
    _dimension(m,"rows");_dimension(n,"columns")
    f=_field(q,field_spec)
    if min(m,n)==1:
        a=[[0]*n for _ in range(m)];a[0][0]=1
        return {"field":f.metadata(),"rows":m,"columns":n,"generators":[a],
                "construction":{"family":"single_matrix_unit"}}
    if min(m,n)==2:
        out=construct_two_row(q,max(m,n),field_spec=f.metadata())
        return out if m==2 else _transpose(out)
    # A diagonal independent family works over every field.
    basis=[]
    for i in range(min(m,n)):
        a=[[0]*n for _ in range(m)];a[i][i]=1;basis.append(a)
    best={"field":f.metadata(),"rows":m,"columns":n,"generators":basis,
          "construction":{"family":"diagonal"}}
    # The all-ones-plus-diagonal construction in square size >=3 gives d+1.
    d=min(m,n)
    j=[[int(i<d and t<d) for t in range(n)] for i in range(m)]
    best["generators"].append(j)
    best["construction"]={"family":"diagonal_plus_all_ones","essential_size":d}
    # Every packaged witness has its own audited field metadata. Only the four
    # supported small field models have packaged arrays; no integer relabeling
    # between different extension-field models is attempted.
    if q not in (2,3,4,5):
        return best
    ledger=_data("result_ledger.json")
    for e in ledger["entries"]:
        if e["q"]!=q or e["m"]<3 or e["n"]<3:
            continue
        for transposed in (False,True):
            a,b=(e["n"],e["m"]) if transposed else (e["m"],e["n"])
            if a>m or b>n or e["lower"]<=len(best["generators"]):
                continue
            name=(f"binary_{e['m']}x{e['n']}_maximum.json" if q==2 else
                  f"q{q}_{e['m']}x{e['n']}_witness.json")
            source=_data(name)
            if transposed:
                source=_transpose(source)
            sf=_field(q,source["field"])
            if sf.metadata()!=f.metadata():
                continue
            padded=[]
            for matrix in source["generators"]:
                padded.append([row+[0]*(n-b) for row in matrix]+[[0]*n for _ in range(m-a)])
            best={"field":f.metadata(),"rows":m,"columns":n,"generators":padded,
                  "construction":{"family":"packaged_verified_witness","source":name,
                                  "transposed":transposed,"padded_from":[a,b],
                                  "verification":"complete_library_and_standalone_field_audits"}}
    return best


def known_results(q,m,n,*,field_spec=None):
    """Return proved bounds. A stored lower bound is not claimed as a search."""
    bound=upper_bound(q,m,n,field_spec=field_spec)
    witness=load_witness(q,m,n,field_spec=field_spec)
    lower=len(witness["generators"])
    upper=bound["upper_bound"]
    if lower>upper:
        raise ArithmeticError("construction and mathematical upper bound disagree")
    return {"schema_version":1,"field":witness["field"],"q":q,"rows":m,"columns":n,
            "status":"EXACT" if lower==upper else "BOUNDED",
            "lower_bound":lower,"upper_bound":upper,"gap":upper-lower,
            "upper_bound_proof":bound,"witness":witness,
            "lower_bound_proof":witness.get("construction"),
            "meaning":"Maximum dimension of an independent rank-one basis with no extra rank-one directions",
            "classification_status":"NOT_INFERRED_FROM_MAXIMUM",
            "novelty":"publication priority not established"}


def _rank_pair_representative(q,m,n,r,s,*,field_spec=None):
    f=_field(q,field_spec)
    def triple(d,rank):
        e1=[1]+[0]*(d-1);e2=[0,1]+[0]*(d-2)
        third=([1,1]+[0]*(d-2)) if rank==2 else [0,0,1]+[0]*(d-3)
        return [e1,e2,third]
    us,vs=triple(m,r),triple(n,s)
    return {"field":f.metadata(),"rows":m,"columns":n,
            "generators":[[[f.mul(x,y) for y in v] for x in u] for u,v in zip(us,vs)],
            "construction":{"family":"complete_k3_class","factor_ranks":[r,s]}}


def known_classes(q,m,n,k=None,*,allow_transpose=False,allow_frobenius=False,field_spec=None):
    """Complete proved small classifications; unknown domains stay explicit.

    k=None returns the supported k=0..maximum rows for two-row shapes, or
    k=0..3 for larger shapes. It does not imply larger k are classified.
    """
    _dimension(m,"rows");_dimension(n,"columns");f=_field(q,field_spec)
    if type(allow_transpose) is not bool or type(allow_frobenius) is not bool:
        raise ValueError("equivalence options must be booleans")
    if allow_transpose and m!=n:
        raise ValueError("transpose equivalence is an option only for square matrices")
    if k is None:
        complete_domain=min(m,n)==1 or min(m,n)==2 and q in (2,3,4,5)
        sizes=(range(2) if min(m,n)==1 else
               range((max(m,n) if max(m,n)<q else q+1)+1)
               if min(m,n)==2 and q in (2,3,4,5) else range(4))
        return {"status":"COMPLETE_CLASSIFICATION" if complete_domain else "PARTIAL_CLASSIFICATION",
                "q":q,"rows":m,"columns":n,
                "scope":("All feasible sizes; larger sizes excluded by the maximum theorem" if complete_domain else
                         "Complete only at the individual sizes explicitly marked complete"),
                "sizes":[known_classes(q,m,n,t,allow_transpose=allow_transpose,
                                      allow_frobenius=allow_frobenius,field_spec=field_spec) for t in sizes]}
    if type(k) is not int or k<0:
        raise ValueError("set size must be a nonnegative integer")
    action=("semilinear" if allow_frobenius else "linear")+"_left_right"+("_with_transpose" if allow_transpose else "")
    reps=[];proof="small_generator_factor_rank_classification"
    if k==0:
        reps=[{"field":f.metadata(),"rows":m,"columns":n,"generators":[]}]
    elif k==1:
        a=[[0]*n for _ in range(m)];a[0][0]=1
        reps=[{"field":f.metadata(),"rows":m,"columns":n,"generators":[a]}]
    elif min(m,n)==1:
        reps=[];proof="one_row_or_column_maximum_one"
    elif min(m,n)==2 and q in (2,3,4,5):
        result=two_row_classes(q,max(m,n),k,field_spec=field_spec)
        reps=result["representatives"]
        if m!=2:
            reps=[_transpose(p) for p in reps]
        proof=result["proof"]
    elif k==2:
        a=[[0]*n for _ in range(m)];a[0][0]=1
        b=[[0]*n for _ in range(m)];b[1][1]=1
        reps=[{"field":f.metadata(),"rows":m,"columns":n,"generators":[a,b]}]
    elif k==3:
        for r,s in ((2,2),(2,3),(3,2),(3,3)):
            if r>m or s>n or ((r,s)==(2,2) and q!=2):
                continue
            if allow_transpose and (r,s)==(3,2):
                continue
            reps.append(_rank_pair_representative(q,m,n,r,s,field_spec=field_spec))
    elif k>upper_bound(q,m,n)["upper_bound"]:
        proof="proved_upper_bound_excludes_this_size"
    else:
        return {"status":"INCONCLUSIVE","q":q,"rows":m,"columns":n,"size":k,
                "action":action,"class_count":None,"representatives":[],
                "reason":"no_complete_classification_for_this_parameter_domain"}
    return {"status":"COMPLETE_CLASSIFICATION","q":q,"rows":m,"columns":n,"size":k,
            "action":action,"class_count":len(reps),"representatives":reps,"proof":proof}
