import json
import queue
import sys

import pytest

from rankone.demo import EXAMPLES, equation, explanation, load_example, matrices_text, parse_matrices, parse_problem, unwrap_problem
from rankone.gui_worker import execute


@pytest.mark.parametrize("index", range(len(EXAMPLES)))
def test_every_preset_round_trips_and_runs_with_expected_decision(index):
    problem = load_example(index)
    assert parse_matrices(matrices_text(problem["generators"])) == problem["generators"]
    f = problem["field"]
    reconstructed = parse_problem(matrices_text(problem["generators"]), str(f["order"]),
                                  json.dumps(f["irreducible_poly"]) if f.get("irreducible_poly") else "")
    bundle = execute({"operation": "check", "problem": reconstructed})
    assert bundle["status"] == ("INVALID" if index in (1, 2) else "VALID")
    text = explanation(bundle)
    assert text.startswith("Field: GF(")
    assert ("PASS:" in text) == (bundle["status"] == "VALID")
    if index == 1:
        assert "A1 + A2 + A3 = R" in text
        assert "[ 2 1 ]" in text and "[ 1 2 ]" in text
        assert bundle["replay"]["verified"] is True
    if index == 3:
        assert "961" in text and "Exactly 20" in text
        assert "maximum dimension >= 20" in text
        assert "Exact maximum" not in text
    if index in (4, 5):
        assert "arithmetic is not modulo q" in text


def test_flexible_plain_text_and_json_input():
    assert parse_matrices("# A1\n1,0\n0,0\n\n# A2\n0 0\n0 1") == [
        [[1, 0], [0, 0]], [[0, 0], [0, 1]]]
    assert parse_matrices("[[[1, 0], [0, 0]]]") == [[[1, 0], [0, 0]]]
    assert equation([0, 1, 2]) == "A2 + 2*A3"


def test_saved_gui_and_cli_reports_import_their_original_matrices():
    problem = load_example(0)
    for data in (problem, {"witness": problem}, {"problem": problem},
                 {"report": {"witness": problem}}):
        assert unwrap_problem(data) == problem
    with pytest.raises(ValueError):
        unwrap_problem({"status": "CANCELLED"})


@pytest.mark.parametrize("text,q,poly", [("", "2", ""), ("# only", "2", ""),
                                         ("x 0", "2", ""), ("1", "two", ""),
                                         ("1", "257", ""), ("1", "4", "[true,1,1]")])
def test_parse_errors_are_actionable(text, q, poly):
    with pytest.raises(ValueError):
        parse_problem(text, q, poly)


def test_input_validation_remains_in_the_real_checker():
    for p in ({"field": {"order": 6}, "generators": [[[1]]]},
              {"field": {"order": 2}, "generators": [[[2]]]},
              {"field": {"order": 2}, "generators": [[[1, 0], [0]]]}):
        with pytest.raises(ValueError):
            execute({"operation": "check", "problem": p})


def test_dependence_and_wrong_rank_get_different_explanations():
    a = [[1, 0], [0, 0]]
    dependent = execute({"operation": "check", "problem": {"field": {"order": 2}, "generators": [a, a]}})
    assert "A1 + A2 = the zero matrix" in explanation(dependent)
    wrong_rank = execute({"operation": "check", "problem": {
        "field": {"order": 2}, "generators": [[[1, 0], [0, 1]]]}})
    assert "A1 has rank 2" in explanation(wrong_rank)


def test_search_witness_is_verified_before_exact_is_presented():
    bundle = execute({"operation": "search", "rows": 5, "columns": 5})
    assert bundle["status"] == "EXACT"
    assert bundle["witness_check"]["status"] == "VALID"
    assert "Exact maximum: 20" in explanation(bundle)
    partial = execute({"operation": "search", "rows": 10, "columns": 10})
    assert partial["status"] == "INCONCLUSIVE"
    assert "Exact maximum" not in explanation(partial)


def test_unfinished_replay_never_gets_pass_label(monkeypatch):
    import rankone.verify as verifier
    monkeypatch.setattr(verifier, "verify", lambda *a, **k: {"verified": None})
    bundle = execute({"operation": "check", "problem": load_example(1)})
    assert bundle["status"] == "INCONCLUSIVE"
    assert "PASS:" not in explanation(bundle)


def test_actual_child_process_returns_a_replayed_failing_example():
    from rankone.gui import ProcessTask
    messages = queue.Queue()
    task = ProcessTask({"operation": "check", "problem": load_example(1)}, messages, 12)
    task.run()
    job_id, bundle = messages.get_nowait()
    assert job_id == 12 and bundle["status"] == "INVALID"
    assert bundle["replay"]["verified"] is True


def test_cancelled_job_never_starts_and_process_deadline_is_inconclusive():
    from rankone.gui import ProcessTask
    messages = queue.Queue()
    task = ProcessTask({"operation": "check", "problem": load_example(0)}, messages, 1)
    task.cancel()
    task.run()
    assert messages.get_nowait()[1]["status"] == "CANCELLED" and task.process is None
    task = ProcessTask({"operation": "check", "problem": load_example(0)}, messages, 2, timeout=0)
    task.run()
    assert messages.get_nowait()[1]["status"] == "INCONCLUSIVE"
    assert task.process.poll() is not None


@pytest.mark.skipif(sys.platform != "darwin", reason="Native Mac Tk smoke test for this delivery")
def test_native_window_clears_stale_results_on_edit_and_loads_search_witness():
    import tkinter as tk
    from rankone.gui import DemoApp
    root = tk.Tk()
    root.withdraw()
    app = DemoApp(root)
    try:
        root.update()
        app.display(execute({"operation": "check", "problem": load_example(0)}))
        root.update()
        assert app.status.cget("text") == "PASS"
        app.q.set("3")
        root.update()
        assert app.bundle is None
        assert str(app.save_button.cget("state")) == "disabled"
        app.display(execute({"operation": "search", "rows": 3, "columns": 3}))
        assert "EXACT" in app.status.cget("text")
        app.load_found()
        root.update()
        assert len(parse_matrices(app.editor.get("1.0", "end"))) == 7
        assert app.bundle is None
    finally:
        app.close()


@pytest.mark.skipif(sys.platform != "darwin", reason="Native Mac Tk interaction and worker smoke test")
def test_native_check_button_runs_real_child_and_saved_result_round_trips(tmp_path, monkeypatch):
    import time
    import tkinter as tk
    from rankone.gui import DemoApp
    root = tk.Tk()
    root.withdraw()
    app = DemoApp(root)
    deadline = time.monotonic() + 8
    try:
        root.update()
        app.actions_menu.invoke(0)
        assert app.task is not None

        def finished():
            if app.task is None or time.monotonic() >= deadline:
                root.quit()
            else:
                root.after(20, finished)

        root.after(20, finished)
        root.mainloop()
        assert app.task is None and app.status.cget("text") == "PASS"
        assert "all 9 possible rank-one directions" in app.saved_text
        result_file = tmp_path / "demo.json"
        monkeypatch.setattr("rankone.gui.filedialog.asksaveasfilename", lambda **kwargs: str(result_file))
        app.save_result()
        assert unwrap_problem(json.loads(result_file.read_text()))["generators"] == load_example(0)["generators"]
    finally:
        app.close()


@pytest.mark.skipif(sys.platform != "darwin", reason="Native Mac Tk layout regression")
@pytest.mark.parametrize("size", ["1180x800", "950x650"])
def test_check_button_stays_inside_visible_window_at_supported_sizes(size):
    import tkinter as tk
    from rankone.gui import DemoApp
    root = tk.Tk()
    app = DemoApp(root)
    try:
        root.geometry(size)
        root.update()
        button = app.check_button
        x = button.winfo_rootx() - root.winfo_rootx()
        y = button.winfo_rooty() - root.winfo_rooty()
        assert button.winfo_ismapped()
        assert button.winfo_width() > 40 and button.winfo_height() > 15
        assert 0 <= x and x + button.winfo_width() <= root.winfo_width()
        assert 0 <= y and y + button.winfo_height() <= root.winfo_height()
        assert app.editor.winfo_height() >= 60
    finally:
        app.close()


@pytest.mark.skipif(sys.platform != "darwin", reason="Native Tk foreground/background contrast")
def test_all_three_text_panels_have_readable_contrast():
    import tkinter as tk
    from rankone.gui import DemoApp
    root = tk.Tk()
    root.withdraw()
    app = DemoApp(root)

    def luminance(widget, color):
        channels = [x / 65535 for x in widget.winfo_rgb(color)]
        linear = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in channels]
        return sum(x * w for x, w in zip(linear, (0.2126, 0.7152, 0.0722)))

    def text_widgets(widget):
        result = [widget] if isinstance(widget, tk.Text) else []
        for child in widget.winfo_children():
            result.extend(text_widgets(child))
        return result

    try:
        panels = text_widgets(root)
        assert len(panels) >= 4
        for panel in panels:
            a = luminance(panel, panel.cget("foreground"))
            b = luminance(panel, panel.cget("background"))
            assert (max(a, b) + 0.05) / (min(a, b) + 0.05) >= 7
    finally:
        app.close()
