"""Plain-language descriptions of completed evidence and remaining limits."""

import json


def _pretty(value):
    return json.dumps(value, indent=2, ensure_ascii=False)


def bound_explanation(proof,q,m,n):
    kind=proof.get('proof')
    c=proof.get('certificate',{})
    if kind=='one_factor_has_only_one_direction':
        return ('With one row or one column, every nonzero matrix has rank one. '
                'Two independent generators would produce an additional rank-one direction when added, so the maximum is one.')
    if kind=='complete_two_row_maximum_theorem':
        other=max(m,n)
        return (f'The complete two-row theorem gives a maximum of {other if other<q else q+1}. '
                f'When the other dimension is smaller than q={q}, the opposite factors must be independent. '
                f'Once that dimension reaches q, the full projective line permits q+1={q+1} generators. '
                'The explicit independent-factor and circuit constructions attain these bounds.')
    if kind=='distinct_factors_and_annihilator_counting':
        lines=[f"The left and right factors must use distinct directions. There are {c.get('left_directions')} left directions and {c.get('right_directions')} right directions.",
               f"Together with the {m*n}-dimensional ambient matrix space, this first gives an upper bound of {c.get('initial_distinct_factor_bound')}."]
        for row in c.get('exclusions',[]):
            k=row['excluded_size'];forced=max(row['forced_directions_left'],row['forced_directions_right'])
            lines += [f"A space of dimension {k} would be defined by {row['codimension']} independent linear equations. Those equations would force at least {forced} rank-one directions into the space.",
                      f"That exceeds the allowed {k} directions. Size {k} is impossible, and a larger good basis would contain a good {k}-element subfamily, so it is impossible too."]
        return '\n'.join(lines)
    if kind=='complete_increasing_prefix_search':
        return 'A complete finite search examined every good family in the declared parameter domain, using the documented safe extension rule.'
    return _pretty(proof)


def research_explanation(bundle):
    if bundle.get("status") == "CANCELLED":
        return "Stopped. This job supplies no completed mathematical decision."
    if bundle.get("error"):
        return "The operation could not finish.\n\n" + str(bundle["error"])
    operation = bundle.get("operation")
    report = bundle.get("report", {})
    status = bundle.get("status", report.get("status", "INCONCLUSIVE"))
    if operation == "check":
        from .demo import explanation
        return explanation(bundle)
    if operation in ("bounds", "search"):
        q = report.get("q", report.get("field", {}).get("order", 2))
        lines = [f"Field F{q}. Matrix size {report.get('rows')} × {report.get('columns')}.", ""]
        lower, upper = report.get("lower_bound", 0), report.get("upper_bound")
        if status == "EXACT":
            lines += [f"Exact maximum: {lower} independent rank-one generators.",
                      "A working construction and a proved upper bound agree."]
        else:
            lines += [f"Known bounds: {lower} ≤ maximum ≤ {upper}.",
                      f"The remaining gap is {upper-lower if upper is not None else 'unknown'}.",
                      "A construction proves the lower bound; the upper-bound argument rules out larger sizes."]
            if status == "INCONCLUSIVE":
                lines += ["This computation reached its limit; the bounds do not constitute a completed maximum claim."]
        proof = report.get("upper_bound_proof", {})
        if proof:
            lines += ["", "Why the upper bound holds:",bound_explanation(proof,q,report.get('rows'),report.get('columns'))]
        if report.get("witness_replay"):
            lines += ["", f"Independent witness replay: {report['witness_replay'].get('verified')}."]
        if operation == "bounds":
            lines += ["", "These are saved research results and mathematical bounds. Choose Check matrices to replay a construction now."]
        else:
            lines += ["", f"Search conclusion: {report.get('reason', '')}.",
                      "A maximum result does not classify all examples attaining it."]
        lines += ["", "Use ‘Load selected matrices’ to inspect the construction."]
        return "\n".join(lines)
    if operation == "table":
        lines = [f"Known maxima and bounds over F{report.get('q')}.",
                 "A single number is exact. An interval is an unresolved gap.", "",
                 "Rows × columns    Maximum / bounds", "────────────────────────────────────"]
        for entry in report.get("entries", []):
            lower, upper = entry["lower_bound"], entry["upper_bound"]
            value = str(lower) if lower == upper else f"{lower}–{upper}"
            lines.append(f"{entry['rows']:>3} × {entry['columns']:<3}          {value}")
        return "\n".join(lines + ["", report.get("scope", ""), "Transposition gives the reversed rectangular sizes."])
    if operation == "construct":
        lines = ["Construction and independent check", ""]
        if report.get("range"):
            lines += ["Construction range:", _pretty(report["range"]), ""]
        if report.get("certificate"):
            lines += ["Construction argument:", _pretty(report["certificate"]), ""]
        if bundle.get("check"):
            lines += [research_explanation(bundle["check"])]
        else:
            lines += [report.get("reason", "No construction is available.")]
        return "\n".join(lines)
    if operation == "equivalence":
        from .demo import matrix_display
        lines = ["Comparing the two spans under the selected coordinate changes.",
                 "Action: " + _pretty(report.get("action", {})), ""]
        if status == "EQUIVALENT":
            c = report["certificate"]
            lines += ["EQUIVALENT: one common row transformation and one common column transformation match every generator direction.",
                      "", "P =", matrix_display(c["P"]), "", "Q =", matrix_display(c["Q"]), "",
                      "Generator matching (A index → B index, starting at 1):",
                      ", ".join(f"{i+1} → {j+1}" for i,j in enumerate(c["permutation"])),
                      f"Nonzero generator scalars: {c['scalars']}",
                      f"Transpose applied to A: {c['transpose']}",
                      f"Common Frobenius power applied to A: {c['frobenius_power']}",
                      "", "The saved certificate records P · transformed(A_i) · Q = scalar_i · B_matching(i).",
                      f"Independent identity replay: {bundle.get('replay', {}).get('verified')}."]
        elif status == "INEQUIVALENT":
            lines += ["INEQUIVALENT under this action.",
                      "The obstruction or completed search is recorded in the saved evidence.",
                      f"Reason: {report.get('reason')}."]
        elif status == "INVALID_INPUT":
            lines += ["One input does not satisfy the good independent-basis condition.",
                      "Check that set first to see its counterexample.", str(report.get("input_reason", report.get("reason", "")))]
        else:
            lines += ["UNFINISHED: no equivalence or inequivalence conclusion is asserted.", str(report.get("reason", ""))]
        return "\n".join(lines)
    if operation == "canonical":
        if status != "COMPLETE":
            return "Canonical form did not finish. No final canonical key is supplied.\n\n" + str(report.get("reason", ""))
        from .demo import matrices_text
        return ("Completed canonical representative under the selected action.\n"
                "The saved result includes its transformation and complete canonical key.\n\n"
                + matrices_text(report["representative"]["generators"]))
    if operation in ("known_classes", "classify", "enumerate_classes"):
        lines = ["Equivalence classes", ""]
        if operation=='classify':
            lines += [f"Classes found among the supplied families: {report.get('class_count') if report.get('class_count') is not None else len(report.get('classes', []))}.",
                      "This groups your supplied examples. It does not exhaust all possible good sets.",
                      f"All supplied inputs processed: {report.get('all_supplied_inputs_processed', False)}."]
        elif status == "COMPLETE_CLASSIFICATION":
            if 'sizes' in report:
                lines += ["Complete classification at each feasible size listed below, under the stated action."]
            else:
                lines += [f"Complete classification for the stated field, shape, size and action: {report.get('class_count', len(report.get('classes', [])))} classes."]
        elif operation == 'enumerate_classes':
            lines += [f"Classes certified so far: {report.get('class_count_lower_bound',len(report.get('classes',[])))}.",
                      "The parameter-domain search did not finish. More classes may exist."]
        else:
            lines += ["A complete classification is not available for this entire requested domain."]
        lines += ["", "Action: " + str(report.get("action", "linear left/right")),
                  "Scope: " + str(report.get("scope", "all good spaces at these parameters"))]
        if report.get("proof"): lines += ["Proof: " + str(report["proof"])]
        for row in report.get("sizes", []):
            lines += [f"Size {row.get('size')}: {row.get('class_count', '?')} classes; {row.get('status')}."]
        for row in report.get("classes", []):
            lines += [f"Class {row.get('class_index', 0)+1}: supplied families " +
                      ", ".join(str(i+1) for i in row.get("input_indices", []))]
        for row in report.get("records", []):
            if row.get("status") != "COMPLETE":
                lines += [f"Family {row.get('input_index', 0)+1}: {row.get('status')} ({row.get('reason')})."]
        if report.get("reason"): lines += ["", str(report["reason"])]
        if report.get("representatives") or report.get("classes"):
            lines += ["", "Choose a representative below, then load its matrices into the checker."]
        return "\n".join(lines)
    if operation == 'factor_analysis':
        lines=['Paired factor ranks and support bounds','']
        if status=='INVALID':
            return 'A generator is not rank one. Check the set first to see the matrix and its rank.'
        lines += [f"Dimension of the matrix span: {report['dimension']}.",
                  f"Independent generators: {report['independent']}.",
                  f"Left-factor span dimension: {report['left_factor_rank']}.",
                  f"Right-factor span dimension: {report['right_factor_rank']}.",
                  f"Subsets checked: {report['subsets_checked']} / {report['subsets_total']}.",
                  f"Mixed supports certified by the rank inequality: {report['mixed_supports_certified']}.",
                  f"Mixed supports left unresolved by this inequality: {report['mixed_supports_unresolved']}.",'']
        if report['sufficient_condition_proved']:
            lines += ['The complete support-rank argument proves the family is good.']
        else:
            lines += ['This sufficient rank test does not determine goodness here. Use Check matrices for the exact decision.']
        lines += ['', 'The paired subset ranks retain common generator labels. They are saved in the result.',
                  'Matching matroid data alone does not prove two spaces equivalent.']
        return '\n'.join(lines)
    if operation == "distribution":
        complete = status == "COMPLETE"
        counts = report.get("matrix_counts" if complete else "partial_matrix_counts", {})
        lines = ["Exact matrix-rank distribution" if complete else "Partial counts — enumeration did not finish", ""]
        lines += [f"Rank {rank}: {count} matrices" for rank, count in counts.items()]
        if complete: lines += ["", f"Total: {report['total_matrices']} matrices in the span."]
        else: lines += ["", "The missing combinations may change these counts."]
        if report.get('proof'):lines += ['',report['proof']]
        return "\n".join(lines)
    if operation == "replay":
        return f"Certificate replay: {status}.\n\n" + _pretty(report.get("replay", {}))
    return _pretty(report or bundle)


def available_problems(bundle):
    """Only actual saved matrix inputs become selectable in the GUI."""
    report = bundle.get("report", {})
    candidates = []
    for label, value in (("Construction", bundle.get("problem")),
                         ("Witness", report.get("witness")),
                         ("Canonical representative", report.get("representative"))):
        if isinstance(value, dict) and "field" in value and "generators" in value:
            candidates.append((label, value))
    for i, value in enumerate(report.get("representatives", [])):
        candidates.append((f"Class {i+1}", value))
    for row in report.get("classes", []):
        candidates.append((f"Class {row['class_index']+1}", row["representative"]))
    for row in report.get("sizes", []):
        for i, value in enumerate(row.get("representatives", [])):
            candidates.append((f"Size {row['size']}, class {i+1}", value))
    return candidates
