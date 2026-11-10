"""Portable desktop launch and menu regression checks."""
import queue
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('failure', [None, OSError('missing taskkill'), subprocess.TimeoutExpired('taskkill', 5)])
def test_windows_stop_includes_descendants_and_falls_back_to_worker_kill(monkeypatch, failure):
    from rankone.gui import ProcessTask
    import rankone.gui as gui
    calls = []
    process = SimpleNamespace(pid=123, poll=lambda: None, kill=lambda: calls.append('kill'))
    def run(args, **kwargs):
        calls.append(args)
        if failure:
            raise failure
    monkeypatch.setattr(gui.sys, 'platform', 'win32')
    monkeypatch.setattr(gui.subprocess, 'run', run)
    task = ProcessTask({}, queue.Queue(), 1)
    task.process = process
    task._kill()
    assert calls == [['taskkill', '/PID', '123', '/T', '/F'], 'kill']


@pytest.mark.skipif(sys.platform not in ('darwin', 'win32'), reason='Native desktop menu regression')
def test_menu_routes_and_current_input_parameters(monkeypatch):
    import tkinter as tk
    from rankone.gui import DemoApp
    root = tk.Tk(); root.withdraw(); app = DemoApp(root)
    requests = []
    monkeypatch.setattr(app, 'start', lambda request: requests.append(request))
    try:
        app.choose_example(4)
        app.use_input_parameters()
        assert (app.search_q.get(), app.rows.get(), app.columns.get(), app.size.get()) == ('4', '2', '2', '')
        app.use_input_parameters(classes=True)
        assert (app.class_q.get(), app.class_m.get(), app.class_n.get(), app.class_k.get()) == ('4', '2', '2', '2')
        app.methods_menu.invoke(2)
        assert requests[-1]['method'] == 'normalized'
        app.run_search_method(1)
        assert requests[-1]['method'] == 'exhaustive'
        assert requests[-1]['q'] == 4
        monkeypatch.setattr('tkinter.simpledialog.askstring', lambda *a, **k: '3, 2, 4, 3')
        app.set_research_parameters()
        assert app.parameter_request('bounds') == dict(operation='bounds', q=3, rows=2, columns=4, k=3)
        app.display({'operation':'check','status':'CANCELLED'})
        app.research_menu.invoke(12)
        assert root.clipboard_get() == app.saved_text
    finally:
        app.close()
