from copy import deepcopy
import json

import pytest

from rankone.checker import check
from rankone.cli import main
from rankone.verify import verify
from rankone.reference import reference_check
from test_reference import E11, E22, J, problem


@pytest.mark.parametrize("q", [2, 3, 4, 5, 8, 9])
@pytest.mark.parametrize("method", ["hybrid", "normalized", "reference"])
def test_certificate_replay(q, method):
    for matrices in ([E11], [E11, E22], [E11, E22, J], [E11, E11], [[[0, 0], [0, 0]]]):
        instance = problem(q, matrices)
        out = reference_check(instance) if method == "reference" else check(instance, method=method)
        assert verify(json.loads(json.dumps(out)))["verified"] is True


def test_modified_witness_is_rejected():
    out = check(problem(3, [E11, E22, J]))
    for change in ("coefficients", "matrix", "u", "status"):
        altered = deepcopy(out)
        if change == "coefficients":
            altered["certificate"][change] = [1, 0, 0]
        elif change == "matrix":
            altered["certificate"][change][0][0] = (altered["certificate"][change][0][0] + 1) % 3
        elif change == "u":
            altered["certificate"][change] = [0, 0]
        else:
            altered["status"] = "VALID"
        assert verify(altered)["verified"] is False


def test_modified_coverage_and_bounds_are_rejected():
    out = check(problem(2, [E11, E22, J]))
    for change in ("missing", "duplicate", "bound", "count", "stats", "kind"):
        altered = deepcopy(out)
        records = altered["certificate"]["supports"]
        if change == "missing":
            records.pop()
        elif change == "duplicate":
            records[1] = deepcopy(records[0])
        elif change == "bound":
            records[0]["rank_u"] = 9
        elif change == "count":
            records[0]["directions"] = 100
        elif change == "stats":
            altered["stats"]["directions_checked"] = 0
        else:
            records[-1]["kind"] = "sylvester"
            records[-1].update(rank_u=2, rank_v=2, lower_bound=1)
        assert verify(altered)["verified"] is False


def test_replay_limit_and_inconclusive_are_not_verified():
    out = check(problem(2, [E11, E22, J]))
    assert verify(out, max_directions=0)["verified"] is None
    assert verify(out, max_supports=0)["verified"] is None
    assert verify(check(problem(2, [E11, E22]), max_supports=0))["verified"] is None


def test_cli_roundtrip_and_exit_codes(tmp_path, capsys):
    source, destination = tmp_path / "input.json", tmp_path / "result.json"
    source.write_text(json.dumps(problem(3, [E11, E22])))
    assert main(["check", str(source), "--output", str(destination), "--text"]) == 0
    assert "Decision: VALID" in capsys.readouterr().out
    assert main(["verify", str(destination)]) == 0
    assert json.loads(capsys.readouterr().out)["verified"] is True
    assert main(["distribution", str(source)]) == 0
    assert json.loads(capsys.readouterr().out)["matrix_counts"] == {"0": 1, "1": 4, "2": 4}
    assert main(["check", str(source), "--max-supports", "0"]) == 2
    capsys.readouterr()
    source.write_text(json.dumps(problem(3, [E11, E22, J])))
    assert main(["check", str(source)]) == 1
    capsys.readouterr()
    source.write_text('{"field":{"order":6},"generators":[[[1]]]}')
    assert main(["check", str(source)]) == 3
    assert "Input error:" in capsys.readouterr().err


def test_extension_model_is_replayed_and_false_metadata_rejected():
    instance = problem(9, [[[1, 3], [3, 4]], E22])
    instance["field"]["irreducible_poly"] = [1, 0, 1]
    out = check(instance)
    assert verify(out)["verified"] is True
    altered = deepcopy(out)
    altered["problem"]["field"]["characteristic"] = 9
    assert verify(altered)["verified"] is False
