"""End-to-end operation, CLI, and native research workflows."""
import json
import os
import queue
import subprocess
import sys
import threading
import time
from copy import deepcopy

import pytest

from rankone.cli import main
from rankone.demo import load_example, parse_matrices
from rankone.operations import execute
from rankone.presentation import available_problems, research_explanation
from rankone.research import load_fixture


@pytest.mark.parametrize('q',[2,3,4,5])
def test_table_does_not_turn_open_bounds_into_exact_results(q):
    bundle=execute(dict(operation='table',q=q))
    entries=bundle['report']['entries']
    assert len(entries)==15
    assert all((r['status']=='EXACT') == (r['lower_bound']==r['upper_bound']) for r in entries)
    square=next(r for r in entries if (r['rows'],r['columns'])==(4,4))
    if q==2:assert square['lower_bound']==square['upper_bound']==12
    else:assert square['lower_bound']==11 and square['upper_bound']==12
    assert 'not an orbit classification' in research_explanation(bundle)


@pytest.mark.parametrize('q,m,n,k',[(3,3,2,2),(4,2,4,5),(5,3,4,6)])
def test_constructors_render_and_replay_actual_basis(q,m,n,k):
    bundle=execute(dict(operation='construct',q=q,rows=m,columns=n,k=k,
                        construction='two_row' if min(m,n)==2 else 'vandermonde',include_infinity=True))
    assert bundle['status']=='VALID'
    assert bundle['check']['replay']['verified'] is True
    assert len(bundle['problem']['generators'])==k
    assert len(available_problems(bundle))==1
    assert 'PASS:' in research_explanation(bundle)


def test_failed_constructor_has_real_counterexample_instead_of_predicted_success():
    b=execute(dict(operation='construct',q=5,rows=2,columns=3,k=4,
                   construction='vandermonde',include_infinity=False))
    assert b['status']=='INVALID'
    assert b['check']['replay']['verified'] is True
    assert 'FAIL:' in research_explanation(b)


def test_equivalence_modes_are_observable_and_certificates_are_exportable():
    a,b=load_fixture('f4_linear_pair_a'),load_fixture('f4_linear_pair_b')
    request=dict(operation='equivalence',problem_a=a,problem_b=b,max_seconds=20)
    linear=execute(request)
    assert linear['status']=='INEQUIVALENT'
    semi=execute({**request,'allow_frobenius':True})
    assert semi['status']=='EQUIVALENT' and semi['replay']['verified'] is True
    assert semi['report']['certificate']['frobenius_power']==1
    replay=execute(dict(operation='replay',data=json.loads(json.dumps(semi))))
    assert replay['status']=='VERIFIED'
    text=research_explanation(semi)
    assert 'P =' in text and 'Q =' in text and 'Frobenius power' in text


def test_partial_and_complete_classification_have_different_scopes():
    p=load_example(0)
    sample=execute(dict(operation='classify',problems=[p,p]))
    assert sample['status']=='PARTIAL_CLASSIFICATION'
    assert 'supplied examples' in research_explanation(sample)
    known=execute(dict(operation='known_classes',q=2,rows=2,columns=3,k=3))
    assert known['status']=='COMPLETE_CLASSIFICATION' and known['report']['class_count']==2
    assert len(available_problems(known))==2
    complete=execute(dict(operation='enumerate_classes',q=2,rows=2,columns=3,k=3,max_candidates=1000000))
    assert complete['status']=='COMPLETE_CLASSIFICATION' and complete['report']['class_count']==2
    partial=execute(dict(operation='enumerate_classes',q=2,rows=2,columns=3,k=3,max_candidates=1))
    assert partial['status']=='PARTIAL_CLASSIFICATION' and partial['report']['class_count'] is None
    assert 'More classes may exist' in research_explanation(partial)


def test_saved_report_tampering_and_interrupted_decisions_never_get_verified():
    p=load_example(0)
    checked=execute(dict(operation='check',problem=p))
    assert execute(dict(operation='replay',data=checked))['status']=='VERIFIED'
    bad=deepcopy(checked);bad['report']['certificate']['fibres'].pop()
    assert execute(dict(operation='replay',data=bad))['status']=='REJECTED'
    limited=execute(dict(operation='check',problem=p,max_candidates=0))
    assert limited['status']=='INCONCLUSIVE'
    assert 'PASS:' not in research_explanation(limited)


def test_analysis_and_exact_low_rank_counts_preserve_their_logical_limits():
    p=load_example(0)
    analysis=execute(dict(operation='factor_analysis',problem=p))
    assert analysis['status']=='COMPLETE'
    assert analysis['report']['sufficient_condition_proved'] is False
    assert 'does not determine goodness' in research_explanation(analysis)
    counts=execute(dict(operation='distribution',problem=p))
    assert counts['status']=='COMPLETE'
    assert counts['report']['method']=='proved_rank_at_most_two_formula'
    assert counts['report']['matrix_counts']=={'0':1,'1':3,'2':4}


def test_new_cli_roundtrips_and_exit_statuses(tmp_path,capsys):
    a=tmp_path/'a.json';b=tmp_path/'b.json';out=tmp_path/'result.json'
    a.write_text(json.dumps(load_example(0)));b.write_text(a.read_text())
    for args in [
        ['check',str(a),'--method','fibre'],
        ['bounds','3','3','3'],
        ['table','4'],
        ['construct','3','2','3','--kind','two_row','--size','4'],
        ['equivalent',str(a),str(b)],
        ['canonical',str(a)],
        ['classes','2','2','3','3'],
        ['enumerate-classes','2','2','2','3'],
        ['factor-analysis',str(a)],
    ]:
        assert main(args+['--output',str(out)])==0
        capsys.readouterr()
        assert json.loads(out.read_text())
    assert main(['bounds','5','4','4'])==2
    capsys.readouterr()


@pytest.mark.skipif(os.name!='posix',reason='POSIX process-group cancellation')
def test_stop_terminates_a_running_job_and_its_replay_worker():
    from rankone.equivalence import canonical_form
    from rankone.gui import ProcessTask
    from rankone.research import load_witness
    small=canonical_form(load_example(0))
    large=load_witness(5,5,5)
    k=len(large['generators'])
    # A syntactically replayable canonical record starts a deliberately large
    # minimum search. Its claim need not be true: this tests stopping replay.
    small.update(problem=large,representative=large,
                 transporter={'P':[[int(i==j) for j in range(5)] for i in range(5)],
                              'Q':[[int(i==j) for j in range(5)] for i in range(5)],
                              'permutation':list(range(k)),'scalars':[1]*k,
                              'transpose':False,'frobenius_power':0})
    messages=queue.Queue()
    task=ProcessTask({'operation':'replay','data':small,'max_candidates':1000000,'max_seconds':120},messages,7,timeout=130)
    thread=threading.Thread(target=task.run,daemon=True);thread.start()
    descendant=None;deadline=time.monotonic()+8
    try:
        while time.monotonic()<deadline:
            if task.process is not None:
                rows=subprocess.run(['ps','-axo','pid=,ppid='],capture_output=True,text=True,check=True).stdout.splitlines()
                for row in rows:
                    pid,parent=map(int,row.split())
                    if parent==task.process.pid:descendant=pid;break
                if descendant is not None:break
            time.sleep(.02)
        assert descendant is not None,'The nested replay worker should have started.'
        task.cancel();thread.join(timeout=3)
        assert not thread.is_alive()
        assert messages.get(timeout=1)[1]['status']=='CANCELLED'
        assert task.process.poll() is not None
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            state=subprocess.run(['ps','-p',str(descendant),'-o','stat='],capture_output=True,text=True).stdout.strip()
            if not state or state.startswith('Z'):break
            time.sleep(.02)
        assert not state or state.startswith('Z'),'Stop must also terminate the replay descendant.'
    finally:
        task.cancel();thread.join(timeout=3)


@pytest.mark.skipif(sys.platform not in ('darwin','win32'),reason='Native desktop GUI workflow')
def test_native_new_views_results_and_worker_are_connected(tmp_path,monkeypatch,isolated_windows_gui):
    if isolated_windows_gui:return
    import tkinter as tk
    from rankone.gui import DemoApp
    r=tk.Tk();r.withdraw();app=DemoApp(r)
    errors=[]
    monkeypatch.setattr('rankone.gui.messagebox.showerror',lambda *a,**k:errors.append(a))

    def finish():
        deadline=time.monotonic()+20
        while app.task is not None and time.monotonic()<deadline:
            r.update();time.sleep(.01)
        r.update()
        assert app.task is None
        assert not errors,errors
        assert app.bundle and app.bundle['status']!='ERROR',app.bundle

    try:
        r.update()
        assert [app.tabs.tab(i,'text') for i in range(5)]==['Check','Bounds / search','Compare','Classes','Explain']
        app.copy_a_to_b();app.compare();finish()
        assert app.bundle['status']=='EQUIVALENT'
        app.canonical();finish()
        assert app.bundle['status']=='COMPLETE'
        app.load_found();assert app.bundle is None
        app.search_q.set('5');app.rows.set('3');app.columns.set('5');app.bounds();finish()
        assert app.bundle['status']=='EXACT'
        app.load_found()
        assert len(parse_matrices(app.editor.get('1.0','end')))==10
        app.class_q.set('2');app.class_m.set('2');app.class_n.set('3');app.class_k.set('3')
        app.known_classes();finish()
        assert len(app.result_problems)==2
        app.result_picker.current(1);app.load_found()
        assert len(parse_matrices(app.editor.get('1.0','end')))==3
        app.add_current();app.add_current();app.classify();finish()
        assert app.bundle['status']=='PARTIAL_CLASSIFICATION'
        assert app.bundle['report']['class_count']==1
        saved=tmp_path/'research-result.json'
        monkeypatch.setattr('rankone.gui.filedialog.asksaveasfilename',lambda **kw:str(saved))
        app.save_result();assert json.loads(saved.read_text())['operation']=='classify'
        app.search_q.set('5');app.rows.set('4');app.columns.set('4');app.bounds();finish()
        assert 'GAP' in app.status.cget('text') and '11 ≤ maximum ≤ 12' in app.saved_text
    finally:app.close()


@pytest.mark.skipif(sys.platform not in ('darwin','win32'),reason='Native controls geometry at supported sizes')
@pytest.mark.parametrize('size',['1280x850','950x650'])
def test_all_tabs_have_accessible_controls_at_supported_window_sizes(size,isolated_windows_gui):
    if isolated_windows_gui:return
    import tkinter as tk
    from rankone.gui import DemoApp
    r=tk.Tk();app=DemoApp(r)
    try:
        r.geometry(size)
        for tab in range(5):
            app.tabs.select(tab);r.update()
            for widget,_ in app.controls:
                if not widget.winfo_ismapped():continue
                x=widget.winfo_rootx()-r.winfo_rootx();y=widget.winfo_rooty()-r.winfo_rooty()
                assert 0<=x and x+widget.winfo_width()<=r.winfo_width(),(tab,widget,x)
                assert 0<=y and y+widget.winfo_height()<=r.winfo_height(),(tab,widget,y)
        assert app.output.winfo_height()>=100
    finally:app.close()
