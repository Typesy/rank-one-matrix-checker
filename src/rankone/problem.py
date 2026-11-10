"""Strict JSON-facing input and result schema."""

from importlib.metadata import version
import platform

from .field import Field
from . import __version__


def prepare(problem):
    if not isinstance(problem, dict) or not isinstance(problem.get("field"), dict):
        raise ValueError("problem needs a field object and a generators list")
    spec = problem["field"]
    f = Field(spec.get("order"), spec.get("irreducible_poly"))
    for key in ("characteristic", "degree", "encoding"):
        if key in spec and spec[key] != f.metadata()[key]:
            raise ValueError(f"field {key} does not match its declared model")
    matrices = problem.get("generators")
    if not isinstance(matrices, list):
        raise ValueError("supply a list of generator matrices")
    declared_shape = None
    if "rows" in problem or "columns" in problem or not matrices:
        rows, columns = problem.get("rows"), problem.get("columns")
        if any(type(x) is not int or x <= 0 for x in (rows, columns)):
            raise ValueError("explicit rows and columns must be positive integers")
        declared_shape = rows, columns
    shape = None
    for matrix in matrices:
        if not isinstance(matrix, list) or not matrix:
            raise ValueError("each matrix needs at least one row")
        if not isinstance(matrix[0], list) or not matrix[0]:
            raise ValueError("each matrix needs at least one column")
        current = len(matrix), len(matrix[0])
        if shape is not None and current != shape:
            raise ValueError("all generators must have the same shape")
        shape = current
        for row in matrix:
            if not isinstance(row, list) or len(row) != shape[1]:
                raise ValueError("matrix rows must be rectangular")
            for value in row:
                f.element(value)
    if declared_shape is not None and shape is not None and shape != declared_shape:
        raise ValueError("declared rows and columns do not match the generators")
    canonical = {"field": f.metadata(), "generators": matrices}
    if declared_shape is not None:
        canonical.update(rows=declared_shape[0], columns=declared_shape[1])
    if "name" in problem:
        if not isinstance(problem["name"], str):
            raise ValueError("name must be a string")
        canonical["name"] = problem["name"]
    return f, matrices, canonical


def result(problem, method, status, reason, certificate, stats):
    return {"schema_version": 1, "problem": problem, "method": method,
            "status": status, "reason": reason, "certificate": certificate,
            "stats": stats,
            "software": {"rankone": __version__, "galois": version("galois"),
                         "numpy": version("numpy"), "python": platform.python_version()}}
