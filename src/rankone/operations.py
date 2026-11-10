"""Shared, bounded operations for GUI workers and other local clients."""

from math import isfinite


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be a whole number from {low} through {high}.")
    return value


def _field(request):
    spec = request.get("field", {"order": request.get("q", 2)})
    if not isinstance(spec, dict):
        raise ValueError("Field information must be an object.")
    _integer(spec.get("order"), "Field order", 2, 256)
    from .field import Field
    metadata=Field(spec["order"], spec.get("irreducible_poly")).metadata()
    for key in ('characteristic','degree','encoding'):
        if key in spec and spec[key] != metadata[key]:
            raise ValueError(f"Field {key} does not match the declared field model.")
    if 'q' in request and request['q'] != metadata['order']:
        raise ValueError("Field order q disagrees with the field model.")
    return metadata


def _limits(request):
    seconds = request.get("max_seconds", 15)
    if type(seconds) not in (int, float) or not isfinite(seconds) or not 0 <= seconds <= 180:
        raise ValueError("Work time must be between 0 and 180 seconds.")
    return {"max_seconds": seconds,
            "max_candidates": _integer(request.get("max_candidates", 100000), "Candidate limit", 0, 1000000)}


def _problem(problem):
    from .problem import prepare
    if not isinstance(problem, dict):
        raise ValueError("Supply a field and generator matrices.")
    _integer(problem.get("field", {}).get("order"), "Field order", 2, 256)
    _, matrices, canonical = prepare(problem)
    m = len(matrices[0]) if matrices else canonical["rows"]
    n = len(matrices[0][0]) if matrices else canonical["columns"]
    if len(matrices) > 100 or m > 10 or n > 10:
        raise ValueError("The interface accepts at most 100 generators and 10 rows or columns.")
    return canonical


def _bundle(operation, report, **extra):
    return {"operation": operation, "status": report["status"], "report": report, **extra}


def execute(request):
    if not isinstance(request, dict):
        raise ValueError("Choose an operation and supply its inputs.")
    operation = request.get("operation", "check")
    limits = _limits(request)
    if operation == "check":
        from .checker import check
        from .verify import verify
        problem = _problem(request.get("problem"))
        method = request.get("method", "fibre")
        if limits['max_seconds']==0:
            from .fibre import fibre_check
            report=fibre_check(problem,max_fibres=limits['max_candidates'],max_seconds=0)
            return _bundle(operation,report,problem=problem,algorithm=report['method'],
                           replay={'verified':None,'reason':'time_limit'})
        if method == "binary_ambient":
            from .binary_membership import binary_membership_check
            report = binary_membership_check(problem, max_candidates=limits["max_candidates"])
            replay=verify(report,max_directions=limits['max_candidates'])
            bundle=_bundle(operation,report,problem=problem,algorithm="binary_ambient",replay=replay)
            if replay['verified'] is not True:bundle['status']='INCONCLUSIVE'
            return bundle
        if method == "reference":
            from .reference import reference_check
            report = reference_check(problem, max_vectors=limits["max_candidates"])
        else:
            report = check(problem, method=method, max_directions=limits["max_candidates"],
                           max_supports=100000, max_seconds=limits["max_seconds"])
        replay = verify(report, max_directions=limits["max_candidates"], max_supports=100000)
        if replay["verified"] is False:
            raise ArithmeticError("Certificate replay rejected the checker result.")
        bundle = _bundle(operation, report, problem=problem, algorithm=report["method"], replay=replay)
        if replay["verified"] is not True:
            bundle["status"] = "INCONCLUSIVE"
            bundle["reason"] = "replay_incomplete"
        return bundle

    if operation == "distribution":
        from .checker import rank_distribution
        problem = _problem(request.get("problem"))
        return _bundle(operation, rank_distribution(problem, max_directions=limits["max_candidates"],
                                                     method='auto',max_seconds=limits['max_seconds']), problem=problem)

    if operation == "factor_analysis":
        from .analysis import factor_analysis
        problem = _problem(request.get("problem"))
        return _bundle(operation, factor_analysis(problem,max_subsets=limits['max_candidates'],
                                                  max_seconds=limits['max_seconds']),problem=problem)

    if operation in ("bounds", "table", "construct", "search", "known_classes", "enumerate_classes"):
        from .research import known_results, load_witness
        field = _field(request)
        q = field["order"]
        m = _integer(request.get("rows", 2), "Rows", 1, 10)
        n = _integer(request.get("columns", 3), "Columns", 1, 10)
        if operation == "table":
            rows = [known_results(q, m1, n1,field_spec=field) for m1 in range(1, 6) for n1 in range(m1, 6)]
            return _bundle(operation, {"status": "COMPLETE", "q": q, "entries": rows,
                                       "scope": "Numerical bounds for 1 <= rows <= columns <= 5; not an orbit classification."})
        if operation == "bounds":
            report = known_results(q, m, n,field_spec=field)
            return _bundle(operation, report)
        k = request.get("k")
        if k is not None:
            _integer(k, "Set size", 0, 100)
        if operation=='enumerate_classes':
            from .classification import classify_parameter_space
            if k is None:raise ValueError("Choose a set size k for the classification domain.")
            report=classify_parameter_space(q,m,n,k,field_spec=field,
                                            allow_transpose=request.get('allow_transpose',False),
                                            allow_frobenius=request.get('allow_frobenius',False),
                                            max_nodes=request.get('max_nodes',50000),**limits)
            return _bundle(operation,report)
        if operation == "known_classes":
            from .research import known_classes
            report = known_classes(q, m, n, k,
                                   allow_transpose=request.get("allow_transpose", False),
                                   allow_frobenius=request.get("allow_frobenius", False),field_spec=field)
            return _bundle(operation, report)
        if operation == "construct":
            from .constructions import construct_two_row, construct_vandermonde
            kind = request.get("construction", "verified")
            if kind == "verified":
                problem = load_witness(q, m, n, field_spec=field)
                if problem is None:
                    return _bundle(operation, {"status": "UNAVAILABLE", "reason": "No packaged witness for these parameters."})
                if k is not None:
                    if k>len(problem['generators']):
                        raise ValueError("The saved construction has fewer generators than the requested size.")
                    problem['generators']=problem['generators'][:k]
                report = {"status": "VALID", "problem": problem,
                          "reason": "Packaged construction; checked again before display."}
            elif kind == "two_row":
                if m != 2 and n != 2:
                    raise ValueError("The two-row construction needs two rows or two columns.")
                problem = construct_two_row(q, n if m == 2 else m, k=k,
                                            variant=request.get("variant", "circuit"), field_spec=field)
                if m != 2:
                    problem["generators"] = [[list(row) for row in zip(*a)] for a in problem["generators"]]
                    problem.update(rows=m,columns=n)
                report = {"status": "VALID", "problem": problem, "reason": "Two-row characterization and explicit construction."}
            elif kind == "vandermonde":
                report = construct_vandermonde(q, m, n, k=k,
                                               include_infinity=bool(request.get("include_infinity", False)),
                                               field_spec=field, parameters=request.get("parameters"))
            else:
                raise ValueError("Choose a verified, two-row, or Vandermonde construction.")
            problem = report["problem"]
            checked = execute({"operation": "check", "problem": problem, **limits})
            # The actual decision takes precedence over constructor prediction.
            return _bundle(operation, report, status=checked["status"], problem=checked["problem"],
                           check=checked, construction_status=report["status"])
        from .search import maximum_search
        report = maximum_search(q, m, n, field_spec=field, method=request.get("method", "greedy"),
                                trials=_integer(request.get("trials", 4), "Trials", 1, 100),
                                seed=_integer(request.get("seed", 20261026), "Seed", 0, 2**32-1),
                                max_nodes=_integer(request.get("max_nodes", 50000), "Node limit", 0, 1000000),
                                use_known=request.get("use_known", True), **limits)
        return _bundle(operation, report)

    if operation in ("equivalence", "canonical", "classify"):
        from .equivalence import equivalent, verify_equivalence, canonical_form, classify_families
        action = {key: request.get(key, False) for key in ("allow_transpose", "allow_frobenius")}
        if any(type(x) is not bool for x in action.values()):
            raise ValueError("Transpose and Frobenius options must be true or false.")
        if operation == "equivalence":
            report = equivalent(_problem(request.get("problem_a")), _problem(request.get("problem_b")), **action, **limits)
            replay = verify_equivalence(report, **limits)
            bundle = _bundle(operation, report, replay=replay)
            if report["status"] == "EQUIVALENT" and replay["verified"] is not True:
                bundle["status"] = "INCONCLUSIVE"
            return bundle
        if operation == "canonical":
            return _bundle(operation, canonical_form(_problem(request.get("problem")), **action, **limits))
        problems = request.get("problems")
        if not isinstance(problems, list) or not 1 <= len(problems) <= 100:
            raise ValueError("Collect between 1 and 100 families for comparison.")
        # User collections are samples, never a certified complete universe.
        report = classify_families([_problem(p) for p in problems], domain_complete=False, **action, **limits)
        return _bundle(operation, report)

    if operation == "replay":
        from .certificates import verify_saved
        data=request.get('data')
        if not isinstance(data,dict):raise ValueError("Load a saved certificate or result.")
        replay=verify_saved(data,**limits)
        status="VERIFIED" if replay.get('verified') is True else "REJECTED" if replay.get('verified') is False else "INCONCLUSIVE"
        return _bundle(operation,{'status':status,'replay':replay})
    raise ValueError("Unknown operation. Choose check, bounds, search, construct, compare, classes, distribution or replay.")
