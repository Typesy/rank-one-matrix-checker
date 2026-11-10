"""Proved rank-one constructions, with explicit finite-field boundary cases."""

from itertools import product

from .field import Field
from .linalg import combine, elimination, factors, flatten


def _dimension(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _field(q, field_spec=None):
    spec = dict(field_spec or {"order": q})
    if spec.get("order", q) != q:
        raise ValueError("field specification order disagrees with q")
    f = Field(q, spec.get("irreducible_poly"))
    for key in ("characteristic", "degree", "encoding"):
        if key in spec and spec[key] != f.metadata()[key]:
            raise ValueError(f"field {key} disagrees with its declared model")
    return f


def projective_vectors(q, dimension):
    """All normalized nonzero vector directions, each exactly once."""
    if type(dimension) is not int or dimension < 0:
        raise ValueError("projective vector dimension must be nonnegative")
    for first in range(dimension):
        for tail in product(range(q), repeat=dimension-first-1):
            yield [0]*first + [1] + list(tail)


def _outer(u, v, field):
    return [[field.mul(x, y) for y in v] for x in u]


def construct_two_row(q, n, k=None, *, variant="circuit", field_spec=None):
    """Construct any feasible two-row good basis.

    At size q+1 the circuit needs n>=q; variant='independent' instead needs
    n>=q+1 and gives the second class. Smaller sizes use independent right
    factors. An empty basis is accepted with explicit shape metadata.
    """
    _dimension(n, "columns")
    f = _field(q, field_spec)
    maximum = n if n < q else q+1
    if k is None:
        k = maximum
    if type(k) is not int or k < 0:
        raise ValueError("set size must be a nonnegative integer")
    if variant not in ("circuit", "independent"):
        raise ValueError("variant must be circuit or independent")
    if k > maximum or (k <= q and k > n):
        raise ValueError("no good two-row basis of this size exists")
    if k == q+1 and variant == "independent" and n < q+1:
        raise ValueError("the independent-factor class needs at least q+1 columns")
    left = list(projective_vectors(q, 2))[:k]
    right = []
    for index in range(k):
        if k == q+1 and index == q and variant == "circuit":
            right.append([1]*q + [0]*(n-q))
        else:
            right.append([int(j == index) for j in range(n)])
    return {"field": f.metadata(), "rows": 2, "columns": n,
            "generators": [_outer(u,v,f) for u,v in zip(left,right)],
            "construction": {"family": "two_row", "variant": variant,
                             "size": k, "proved_maximum": maximum,
                             "proof": "two_row_characterization",
                             "right_factor_rank": min(k,q) if variant == "circuit" and k == q+1 else k}}


def _w_vector(t, d, field):
    if t is None:
        return [0]*(d-1)+[1]
    vector, value = [], 1
    for _ in range(d):
        vector.append(value)
        value = field.mul(value,t)
    return vector


def construct_vandermonde(q, m, n, parameters=None, *, include_infinity=False,
                          k=None, field_spec=None):
    """Return a construction report, including invalid boundary certificates.

    Parameters are field labels, with None or 'infinity' denoting infinity.
    Default selects all permitted parameters; k chooses a prefix. m,n>=2.
    The report's problem is usable even on invalid ranges for demonstration.
    """
    _dimension(m, "rows")
    _dimension(n, "columns")
    if min(m,n) < 2:
        raise ValueError("the Vandermonde family requires at least two rows and columns")
    f = _field(q, field_spec)
    if parameters is not None and k is not None:
        raise ValueError("supply parameters or k, not both")
    if parameters is None:
        parameters = list(range(q)) + ([None] if include_infinity else [])
        if k is not None:
            if type(k) is not int or not 0 <= k <= len(parameters):
                raise ValueError("k exceeds the number of selected field parameters")
            parameters = parameters[:k]
    if not isinstance(parameters, (list,tuple)):
        raise ValueError("parameters must be a list of distinct field labels or infinity")
    selected = []
    for t in parameters:
        if t is None or t == "infinity":
            t = None
        else:
            f.element(t)
        if t in selected:
            raise ValueError("Vandermonde parameters must be distinct")
        selected.append(t)
    matrices = [_outer(_w_vector(t,m,f),_w_vector(t,n,f),f) for t in selected]
    k, d = len(matrices), m+n-1
    good = k <= d-1 or k == q+1 == d
    problem = {"field": f.metadata(), "rows": m, "columns": n,
               "generators": matrices,
               "construction": {"family":"extended_vandermonde",
                                "parameters": selected}}
    certificate = {"proof": "exact_rank_one_hankel_range",
                   "hankel_dimension": d, "parameters": selected}
    reason = "support_rank_bound" if k <= d-1 else "complete_projective_line_boundary"
    if not good and k > d:
        dimension, relation = elimination([flatten(a) for a in matrices], f, track=True)
        assert dimension < k and relation is not None
        certificate.update(coefficients=relation, dimension=dimension)
        reason = "dependent_generators"
    elif not good:
        # k=d independent generators span the Hankel space. Any missing point
        # is an explicit extra rank-one direction. Solve its coordinates using
        # the field library, independently of the later checker.
        missing = next(t for t in list(range(q))+[None] if t not in selected)
        extra = _outer(_w_vector(missing,m,f),_w_vector(missing,n,f),f)
        import numpy as np
        augmented = f.GF([[*row] for row in zip(*[flatten(a) for a in matrices])])
        augmented = np.concatenate((augmented, f.GF(flatten(extra))[:,None]), axis=1)
        reduced = augmented.row_reduce()
        coefficients = [0]*k
        for row in reduced.tolist():
            pivot = next((j for j,x in enumerate(row[:k]) if x),None)
            if pivot is not None:
                coefficients[pivot] = row[k]
        assert combine(matrices,coefficients,f) == extra
        u,v = factors(extra,f)
        certificate.update(coefficients=coefficients,matrix=extra,u=u,v=v,
                           missing_parameter=missing)
        reason = "extra_rank_one_missing_parameter"
    return {"status":"VALID" if good else "INVALID", "reason":reason,
            "problem":problem, "certificate":certificate,
            "range":{"finite_maximum":min(q,d-1),
                     "extended_maximum":q+1 if q+1<=d-1 or q+1==d else d-1,
                     "condition":"k<=m+n-2 or k=q+1=m+n-1"}}


def two_row_classes(q, n, k, *, field_spec=None):
    """Complete explicit good-space classes for q<=5 in a two-row shape."""
    _dimension(n,"columns")
    _field(q,field_spec)
    if q not in (2,3,4,5):
        return {"status":"INCONCLUSIVE", "reason":"projective_line_subset_orbits_not_classified",
                "representatives":[], "class_count":None}
    if type(k) is not int or k < 0:
        raise ValueError("set size must be a nonnegative integer")
    representatives=[]
    if k == 0 or 1 <= k <= min(n,q):
        representatives=[construct_two_row(q,n,k,field_spec=field_spec)]
    elif k == q+1 and n >= q:
        representatives=[construct_two_row(q,n,k,variant="circuit",field_spec=field_spec)]
        if n >= q+1:
            representatives.append(construct_two_row(q,n,k,variant="independent",field_spec=field_spec))
    return {"status":"COMPLETE_CLASSIFICATION", "q":q,"rows":2,"columns":n,"size":k,
            "action":"linear_left_right", "class_count":len(representatives),
            "representatives":representatives,
            "proof":"two_row_characterization_and_PGL2_subset_transitivity_for_q_at_most_5"}
