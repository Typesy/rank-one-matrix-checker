"""Input and explanation helpers for the local demo, independent of Tk."""

import json
import re
from pathlib import Path

EXAMPLES = (
    ("Three generators over F2 — passes", "triple_f2.json"),
    ("Same generators over F3 — fails", "triple_f3.json"),
    ("Shared factor over F5 — fails", "shared_column_f5.json"),
    ("Twenty generators, 5 by 5 over F2", "binary_5x5_maximum.json"),
    ("Extension field F4 — passes", "extension_f4.json"),
    ("Extension field F9 — passes", "extension_f9.json"),
    ("Six generators, 3 by 3 over F3", "@witness:3:3:3"),
    ("Eight generators, 3 by 4 over F4", "@witness:4:3:4"),
    ("Ten generators, 3 by 5 over F5", "@witness:5:3:5"),
    ("Eighteen generators, 5 by 5 over F5", "@witness:5:5:5"),
    ("F5 equivalence example — matching one", "@fixture:f5_mds_identity"),
    ("F5 equivalence example — matching two", "@fixture:f5_mds_transposition"),
    ("F4 linear / semilinear example — A", "@fixture:f4_linear_pair_a"),
    ("F4 linear / semilinear example — B", "@fixture:f4_linear_pair_b"),
)

TALK = """Our question is how many independent rank-one matrices can generate a space
without producing any additional rank-one directions. We fix the field and
matrix size, and vary the number of generators. Scalar multiples count as
one direction. The number of independent generators is the space dimension.

The checker starts with the actual matrices. It verifies that each has rank
one and that the whole matrices are independent. It then describes their span
by linear equations. To find rank-one members, it writes a candidate as uv^T.
For each possible left-factor direction u, those equations become a small
linear system for the right factor v. The solution kernel lists every allowed
right direction. Every rank-one matrix has such a factorization, so finishing
all left directions covers every possibility. A separate implementation
replays the equations and kernel dimensions.

A passing set has exactly its supplied rank-one directions. A failing set
comes with an explicit dependence, wrong-rank generator, or extra rank-one
matrix and coefficients. For example E11, E22 and the all-ones 2 by 2 matrix
pass over F2. Over F3 the same set fails: adding all three gives [[2,1],[1,2]],
which has rank one and is a fourth direction. The checker may display a
nonzero scalar multiple of that counterexample; it proves the same failure.

Bounds and search answer a different part of the research question. A working
set of size k proves a lower bound. An argument excluding every larger set
proves an upper bound. Equal bounds establish the maximum. An unsuccessful
search gives no new upper bound. Greedy searches find constructions; complete
small searches can exclude larger possibilities. Every reported search
witness is checked and replayed before its lower bound is used.

Equivalence asks whether one common invertible row change and one common
invertible column change carry all generator directions onto the other set.
The output supplies those transformations and the generator matching. Transpose
for square matrices and a common field automorphism are explicit options.
Over F4 the latter is x -> x^2. These options change the acting group and may
merge classes. Matching rank distributions or matroids alone cannot prove
equivalence. Canonical forms are supplied only when their full search finishes.

Classes reports complete known small domains or groups the examples we supply.
Grouping our collection does not establish that all possible examples were
found. Work limits and Stop preserve that distinction. Saved JSON contains the
matrices, field model, transformations or proof evidence, and completion status.
Extension-field arithmetic uses an explicit polynomial model; F4 is not
integer arithmetic modulo four. Our verified computations establish their
stated finite claims; publication novelty and broader results need separate work.
"""


def examples_directory():
    return Path(__file__).resolve().parents[2] / "examples"


def load_example(index):
    name = EXAMPLES[index][1]
    if name.startswith('@witness:'):
        from .research import load_witness
        return load_witness(*map(int, name.split(':')[1:]))
    if name.startswith('@fixture:'):
        from .research import load_fixture
        return load_fixture(name.split(':')[1])
    path = examples_directory() / name
    if not path.exists():
        path = Path(__file__).parent / 'data' / ('demo_' + name)
    return json.loads(path.read_text())


def unwrap_problem(data):
    """Import a normal input, a CLI witness, or a result saved by this demo."""
    if isinstance(data, dict):
        if "field" in data and "generators" in data:
            return data
        for path in (("witness",), ("problem",), ("report", "witness")):
            candidate = data
            for key in path:
                candidate = candidate.get(key) if isinstance(candidate, dict) else None
            if isinstance(candidate, dict) and "field" in candidate and "generators" in candidate:
                return candidate
    raise ValueError("This JSON file does not contain field information and generator matrices.")


def matrices_text(matrices):
    return "\n\n".join(f"# A{i + 1}\n" + "\n".join(
        " ".join(str(x) for x in row) for row in a)
        for i, a in enumerate(matrices))


def parse_matrices(text):
    if len(text) > 200000:
        raise ValueError("Please use at most 200,000 characters in this demo.")
    text = text.strip()
    if not text:
        raise ValueError("Enter at least one matrix or load an example.")
    if text.startswith("["):
        return json.loads(text)
    matrices = []
    for block in re.split(r"\n\s*\n", text):
        rows = []
        for line in block.splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                try:
                    rows.append([int(x) for x in re.split(r"[\s,]+", line)])
                except ValueError as exc:
                    raise ValueError("Matrix entries must be integers. Separate matrices by a blank line.") from exc
        if rows:
            matrices.append(rows)
    if not matrices:
        raise ValueError("Enter matrix rows as numbers, not only comments.")
    return matrices


def parse_problem(text, field_text, polynomial_text=""):
    try:
        order = int(field_text)
    except ValueError as exc:
        raise ValueError("Field order q must be a whole number, such as 2 or 5.") from exc
    if not 2 <= order <= 256:
        raise ValueError("This demo accepts field orders from 2 through 256. The order must be a prime power.")
    field = {"order": order}
    polynomial_text = polynomial_text.strip()
    if polynomial_text:
        try:
            poly = json.loads(polynomial_text) if polynomial_text.startswith("[") else [
                int(x) for x in re.split(r"[\s,]+", polynomial_text)]
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Use polynomial coefficients such as 1,1,1 (highest degree first).") from exc
        if not isinstance(poly, list) or any(type(x) is not int for x in poly):
            raise ValueError("Polynomial coefficients must be a list of integers.")
        field["irreducible_poly"] = poly
    return {"field": field, "generators": parse_matrices(text)}


def matrix_display(matrix):
    width = max(len(str(x)) for row in matrix for x in row)
    return "\n".join("[ " + " ".join(f"{x:>{width}}" for x in row) + " ]" for row in matrix)


def equation(coefficients):
    terms = [f"A{i + 1}" if x == 1 else f"{x}*A{i + 1}"
             for i, x in enumerate(coefficients) if x]
    return " + ".join(terms) or "0"


def explanation(bundle):
    """Describe actual report evidence, never upgrade an unfinished computation."""
    if bundle.get('operation') not in ('check', 'search') or (bundle.get('operation') == 'search' and 'field' in bundle.get('report', {})):
        from .presentation import research_explanation
        return research_explanation(bundle)
    if bundle.get("error"):
        return "The input could not be checked.\n\n" + bundle["error"]
    report = bundle["report"]
    if bundle["operation"] == "search":
        lower, upper = report["lower_bound"], report["upper_bound"]
        lines = [f"Binary matrices: {report['rows']} rows by {report['columns']} columns.", ""]
        if bundle["status"] == "EXACT":
            lines += [f"Exact maximum: {lower} independent generators.",
                      f"A construction of size {lower} passed the separate membership checker.",
                      f"The upper bound is also {upper}.",
                      "The bounds agree, so this size is the maximum."]
        else:
            lines += ["The search or independent verification did not finish.",
                      f"Reported search bounds: {lower} <= maximum <= {upper}.",
                      "This is not a completed maximum claim."]
        witness_check = bundle.get("witness_check")
        if witness_check:
            lines += ["", f"Witness verification: {witness_check['status']}.",
                      f"Ambient rank-one candidates checked: {witness_check['candidates_examined']} / {witness_check['candidates_total']}."]
        if lower:
            lines += ["", "Use ‘Load found matrices’ to inspect or check the saved construction."]
        return "\n".join(lines)

    problem = bundle["problem"]
    a, field = problem["generators"], problem["field"]
    q, k = field["order"], len(a)
    m = len(a[0]) if a else problem['rows']
    n = len(a[0][0]) if a else problem['columns']
    lines = [f"Field: GF({q}). Matrices: {m} by {n}. Generators: {k}.", ""]
    cert = report.get("counterexample") or report.get("certificate", {})
    if bundle["status"] == "VALID":
        lines += ["PASS: the supplied matrices are an independent rank-one basis",
                  "with no extra rank-one directions in their span.", ""]
        if bundle["algorithm"] == "binary_ambient":
            lines += [f"All {k} generators have rank one and are independent.",
                      f"Checked all {report['candidates_total']} possible rank-one matrices of this size.",
                      f"Exactly {len(report['rank_one_members'])} belong to the span: the supplied generators."]
        elif bundle['algorithm'] == 'annihilator-fibre':
            stats = report['stats']
            lines += [f"All {k} generators have rank one and are independent.",
                      f"Checked {stats['fibres_checked']} / {stats['fibres_total']} projective factor directions.",
                      f"Their right-factor kernels cover all {stats['ambient_rank_one_directions']} possible rank-one directions of this size.",
                      f"Exactly {stats['rank_one_directions']} belong to the span: the supplied generator directions.",
                      "A separate arithmetic and elimination replay confirmed the complete certificate."]
        elif bundle['algorithm'] == 'reference-full':
            lines += [f"Mixed coefficient vectors checked: {report['stats']['vectors_checked']}.",
                      "The complete coefficient list was independently replayed."]
        else:
            stats = report["stats"]
            lines += [f"Mixed directions checked directly: {stats['directions_checked']}.",
                      f"Mixed directions excluded by the proved rank bound: {stats['directions_by_bound']}.",
                      "A separate Gaussian-rank replay confirmed this result."]
        lines += ["", f"This construction gives a lower bound: maximum dimension >= {k}.",
                  "A maximum claim also needs a matching upper-bound argument."]
    elif bundle["status"] == "INVALID":
        reason = report["reason"]
        if reason == "extra_rank_one":
            lines += ["FAIL: a combination creates an extra rank-one matrix.", "",
                      equation(cert["coefficients"]) + " = R", "", matrix_display(cert["matrix"]), "",
                      "R has rank one and is not one of the supplied generator directions.",
                      "It belongs to the span because of the displayed combination."]
            if "u" in cert:
                lines += ["", f"Rank-one factorization: u = {cert['u']}", f"                         v = {cert['v']}",
                          "R = u v^T, with arithmetic in the stated field."]
        elif reason == "dependent_generators":
            lines += ["FAIL: the proposed generators are dependent.", "",
                      equation(cert["coefficients"]) + " = the zero matrix.",
                      "At least one matrix can be generated from the others.",
                      "This is a basis failure, distinct from finding an extra rank-one direction."]
        else:
            index = cert["index"]
            lines += [f"FAIL: A{index + 1} has rank {cert['rank']}, so it is not rank one."]
    else:
        lines += ["UNFINISHED: a computation or replay reached its limit.",
                  "No passing result is asserted.", f"Reason: {bundle.get('reason', report.get('reason', 'limit'))}."]
    if field.get("degree", 1) > 1:
        lines += ["", f"Extension-field polynomial (highest degree first): {field['irreducible_poly']}.",
                  "Entry labels represent polynomial-basis field elements; arithmetic is not modulo q."]
    return "\n".join(lines)
