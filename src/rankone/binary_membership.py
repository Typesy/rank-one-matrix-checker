"""Independent binary witness checking by ambient rank-one membership.

Gaussian reduction comes from the finite-field library; no search routines,
incremental span enumeration, or optimized rank-one predicate are used here.
Every nonzero binary rank-one matrix has exactly one factorization uv^T with
nonzero u,v. Thus testing the whole factor grid is an exhaustive check, often
much smaller than enumerating all 2**k coefficient vectors. This is exact
finite computation, not a formally verified proof kernel.
"""

from .problem import prepare


def _bits(entries):
    return sum(int(value) << i for i, value in enumerate(entries))


def _matrix(value, m, n):
    return [[(value >> (i * n + j)) & 1 for j in range(n)] for i in range(m)]


def binary_membership_check(problem, *, max_candidates=100000):
    """Check a supplied independent rank-one basis over F_2.

Returns VALID only after all ambient rank-one matrices have been examined.
An INVALID extra-rank-one result includes the matrix and coefficients in the
supplied basis; an INCONCLUSIVE result contains only the members seen so far.
The candidate budget excludes initial input validation and Gaussian reduction.
"""
    if type(max_candidates) is not int or max_candidates < 0:
        raise ValueError("max_candidates must be a nonnegative integer")
    field, matrices, canonical = prepare(problem)
    if field.q != 2:
        raise ValueError("binary membership checking requires field order 2")
    k = len(matrices)
    m = len(matrices[0]) if k else canonical['rows']
    n = len(matrices[0][0]) if k else canonical['columns']
    width = m * n
    rows = [[x for row in matrix for x in row] for matrix in matrices]
    reduced = field.GF(rows).row_reduce().tolist() if rows else []
    dimension = sum(any(row) for row in reduced)
    total = ((1 << m) - 1) * ((1 << n) - 1)
    examined, members = 0, []

    def finish(status, reason, counterexample=None):
        return {
            "schema_version": 1,
            "method": "binary_ambient_membership",
            "status": status,
            "reason": reason,
            "dimension": dimension,
            "generator_count": k,
            "candidates_examined": examined,
            "candidates_total": total,
            "rank_one_members": members,
            "counterexample": counterexample,
            "problem": canonical,
            "limits": {"max_candidates": max_candidates},
        }

    for index, matrix in enumerate(matrices):
        rref = field.GF(matrix).row_reduce().tolist()
        matrix_rank = sum(any(row) for row in rref)
        if matrix_rank != 1:
            return finish("INVALID", "generator_not_rank_one",
                          {"index": index, "rank": matrix_rank})
    if dimension != k:
        relation = field.GF(rows).left_null_space()[0].tolist()
        return finish("INVALID", "dependent_generators", {"coefficients": relation})

    # The augmented identity records how each reduced row depends on the input
    # basis. Independence guarantees every pivot is in the matrix-entry block.
    augmented = [row + [int(i == j) for j in range(k)] for i, row in enumerate(rows)]
    transformed = field.GF(augmented).row_reduce().tolist() if augmented else []
    pivots = {}
    for row in transformed:
        value = _bits(row[:width])
        first = (value & -value).bit_length() - 1
        pivots[first] = (value, _bits(row[width:]))
    supplied = {_bits(row) for row in rows}

    for u in range(1, 1 << m):
        for v in range(1, 1 << n):
            if examined >= max_candidates:
                return finish("INCONCLUSIVE", "candidate_limit")
            # Expand the outer product directly, independently of binary_search.
            entries = [((u >> i) & 1) * ((v >> j) & 1)
                       for i in range(m) for j in range(n)]
            candidate = _bits(entries)
            remainder, coefficients = candidate, 0
            while remainder:
                first = (remainder & -remainder).bit_length() - 1
                if first not in pivots:
                    break
                row, combination = pivots[first]
                remainder ^= row
                coefficients ^= combination
            examined += 1
            if remainder:
                continue
            matrix = _matrix(candidate, m, n)
            members.append(matrix)
            if candidate not in supplied:
                return finish("INVALID", "extra_rank_one", {
                    "matrix": matrix,
                    "coefficients": [(coefficients >> i) & 1 for i in range(k)],
                    "u": [(u >> i) & 1 for i in range(m)],
                    "v": [(v >> j) & 1 for j in range(n)],
                })
    return finish("VALID", "all_ambient_rank_one_matrices_checked")


def verify_binary_membership(report, *, max_candidates=100000):
    """Replay old binary reports with a different complete census method."""
    if type(max_candidates) is not int or max_candidates<0:
        raise ValueError("replay limit must be a nonnegative integer")
    try:
        from .linalg import flatten,rank,rank_one
        from .fibre import fibre_check,verify_fibre
        if (type(report['schema_version']) is not int or report['schema_version']!=1
                or report['method']!='binary_ambient_membership'):
            return {'verified':False,'reason':'unknown_binary_schema'}
        f,a,p=prepare(report['problem'])
        if f.q!=2:return {'verified':False,'reason':'not_binary'}
        if report['status']=='INCONCLUSIVE':return {'verified':None,'reason':'no_mathematical_decision'}
        k=len(a);m=len(a[0]) if k else p['rows'];n=len(a[0][0]) if k else p['columns']
        dimension=rank([flatten(matrix) for matrix in a],f)
        if (type(report['dimension']) is not int or report['dimension']!=dimension or
                type(report['generator_count']) is not int or report['generator_count']!=k):
            return {'verified':False,'reason':'incorrect_dimension'}
        if report['status']=='INVALID':
            from .verify import verify
            converted={'schema_version':1,'method':'normalized','problem':p,
                       'status':'INVALID','reason':report['reason'],
                       'certificate':report['counterexample'],'stats':{}}
            return verify(converted,max_directions=max_candidates)
        if report['status']!='VALID' or report['reason']!='all_ambient_rank_one_matrices_checked':
            return {'verified':False,'reason':'unknown_binary_claim'}
        total=((1<<m)-1)*((1<<n)-1)
        if any(type(report[key]) is not int or report[key]!=total for key in ('candidates_total','candidates_examined')):
            return {'verified':False,'reason':'incorrect_ambient_coverage'}
        if dimension!=k or any(not rank_one(matrix,f) for matrix in a):
            return {'verified':False,'reason':'invalid_basis'}
        members=report['rank_one_members']
        if not isinstance(members,list) or len(members)!=k:
            return {'verified':False,'reason':'incorrect_member_list'}
        # Exact binary entries, shapes and set membership are checked by prepare.
        if members:
            prepare({'field':f.metadata(),'rows':m,'columns':n,'generators':members})
        if sorted(tuple(flatten(x)) for x in members)!=sorted(tuple(flatten(x)) for x in a):
            return {'verified':False,'reason':'incorrect_member_list'}
        checked=fibre_check(p,max_fibres=max_candidates)
        if checked['status']=='INCONCLUSIVE':return {'verified':None,'reason':'replay_limit'}
        if checked['status']!='VALID':return {'verified':False,'reason':'extra_direction_found'}
        replay=verify_fibre(checked,max_fibres=max_candidates)
        return {'verified':replay['verified'],'reason':'binary_decision_replayed_by_fibres'}
    except (ValueError,TypeError,KeyError,IndexError,AttributeError,ZeroDivisionError):
        return {'verified':False,'reason':'malformed_or_false_certificate'}
