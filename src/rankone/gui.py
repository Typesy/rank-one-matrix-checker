"""Native local demonstration window. Run with python -m rankone.gui."""

import json
import os
import queue
import subprocess
import signal
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from .demo import EXAMPLES, TALK, explanation, load_example, matrices_text, parse_problem, unwrap_problem
from .presentation import available_problems

TEXT_COLORS = {"background": "#fcfcfe", "foreground": "#172233",
               "insertbackground": "#172233", "selectbackground": "#c4d8f8",
               "selectforeground": "#10233d"}


class ProcessTask:
    """Keep computation outside Tk, with cancellation and a hard process limit."""

    def __init__(self, request, messages, job_id, timeout=20):
        self.request, self.messages, self.job_id = request, messages, job_id
        self.timeout = timeout
        self.cancelled = threading.Event()
        self.process = None

    def cancel(self):
        self.cancelled.set()
        if self.process is not None and self.process.poll() is None:
            self._kill()

    def _kill(self):
        # Replay may have its own deadline worker. Stop the whole job group.
        try:
            if sys.platform == "win32":
                # Windows has no POSIX process groups. Include replay workers.
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                        capture_output=True, timeout=5, check=False,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    pass
                finally:
                    if self.process.poll() is None:
                        self.process.kill()
            elif os.name == "posix":
                os.killpg(self.process.pid, signal.SIGKILL)
            else:
                self.process.kill()
        except ProcessLookupError:
            pass

    def run(self):
        try:
            if self.cancelled.is_set():
                self.messages.put((self.job_id, {"status": "CANCELLED"}))
                return
            self.process = subprocess.Popen(
                [sys.executable, "-m", "rankone.gui_worker"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True,
                start_new_session=(os.name=='posix'),
            )
            started, first = time.monotonic(), True
            while True:
                if self.cancelled.is_set():
                    self._kill()
                    self.process.communicate()
                    self.messages.put((self.job_id, {"status": "CANCELLED"}))
                    return
                if time.monotonic() - started >= self.timeout:
                    self._kill()
                    self.process.communicate()
                    self.messages.put((self.job_id, {"status": "INCONCLUSIVE", "error":
                        f"The job reached its {self.timeout:g}-second process limit. No completed decision is asserted."}))
                    return
                try:
                    out, err = self.process.communicate(
                        json.dumps(self.request) if first else None, timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    first = False
            if self.cancelled.is_set():
                result = {"status": "CANCELLED"}
            elif self.process.returncode:
                result = {"status": "ERROR", "error": err[-3000:] or "The worker stopped unexpectedly."}
            else:
                result = json.loads(out)
            self.messages.put((self.job_id, result))
        except Exception as exc:
            self.messages.put((self.job_id, {"status": "ERROR", "error": str(exc)}))


class DemoApp:
    def __init__(self, root):
        self.root = root
        root.title("Rank-one matrix research")
        root.geometry("1280x850")
        root.minsize(950, 650)
        root.configure(background="#f5f6fa")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.messages, self.job_id, self.task = queue.Queue(), 0, None
        self.bundle, self.saved_text, self.loading = None, "", False
        self.collection, self.result_problems = [], []
        self.empty_shape = None
        style = ttk.Style(root)
        style.configure("Title.TLabel", font=("Helvetica", 23, "bold"))
        style.configure("Sub.TLabel", font=("Helvetica", 12))
        style.configure("Section.TLabel", font=("Helvetica", 14, "bold"))

        header = ttk.Frame(root, padding=(24, 20, 24, 14))
        header.pack(fill="x")
        ttk.Label(header, text="Rank-one matrix research", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Check good sets, investigate maxima, and compare equivalence classes over finite fields.",
                  style="Sub.TLabel").pack(anchor="w", pady=(5, 0))
        pane = ttk.Panedwindow(root, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        left = ttk.Frame(pane, padding=10)
        right = ttk.Frame(pane, padding=14)
        pane.add(left, weight=1)
        pane.add(right, weight=1)
        self.tabs = ttk.Notebook(left)
        self.tabs.pack(fill="both", expand=True)
        check_tab, search_tab, compare_tab, classes_tab, talk_tab = (ttk.Frame(self.tabs, padding=10) for _ in range(5))
        for tab, label in zip((check_tab, search_tab, compare_tab, classes_tab, talk_tab),
                              ("Check", "Bounds / search", "Compare", "Classes", "Explain")):
            self.tabs.add(tab, text=label)
        self.controls = []
        self._check_tab(check_tab)
        self._search_tab(search_tab)
        self._compare_tab(compare_tab)
        self._classes_tab(classes_tab)
        self._talk_tab(talk_tab)

        self.status = tk.Label(right, text="READY", anchor="w", font=("Helvetica", 17, "bold"),
                               background="#e9edf5", foreground="#34435b", padx=12, pady=10)
        self.status.pack(fill="x", pady=(0, 10))
        ttk.Label(right, text="Result and mathematical evidence", style="Section.TLabel").pack(anchor="w")
        # Reserve action space before allowing the text area to fill the rest.
        actions = ttk.Frame(right)
        actions.pack(side="bottom", fill="x")
        buttons = ttk.Frame(actions)
        buttons.pack(fill="x")
        self.copy_button = ttk.Button(buttons, text="Copy explanation", command=self.copy_explanation, state="disabled")
        self.copy_button.pack(side="left")
        self.save_button = ttk.Button(buttons, text="Save result…", command=self.save_result, state="disabled")
        self.save_button.pack(side="left", padx=6)
        self.cancel_button = ttk.Button(buttons, text="Stop", command=self.cancel, state="disabled")
        self.cancel_button.pack(side="right")
        self.found_button = ttk.Button(actions, text="Load found matrices into checker", command=self.load_found, state="disabled")
        self.found_button.pack(anchor="w", pady=(0, 6))
        self.result_picker = ttk.Combobox(actions, state="disabled", width=42)
        self.result_picker.pack(fill="x", pady=(0, 6))
        limit_bar = ttk.Frame(actions)
        limit_bar.pack(fill="x", pady=(0, 6))
        ttk.Label(limit_bar, text="Work limit (seconds)").pack(side="left")
        self.work_seconds = tk.StringVar(value="15")
        entry = ttk.Spinbox(limit_bar, from_=1, to=180, textvariable=self.work_seconds, width=5)
        entry.pack(side="left", padx=6)
        self.controls.append((entry, "normal"))
        self._button(limit_bar, "Replay saved certificate…", self.replay_file).pack(side="right")
        self.output = scrolledtext.ScrolledText(right, wrap="word", font=("Menlo" if sys.platform == "darwin" else "Consolas", 12),
                                               relief="flat", padx=12, pady=12, **TEXT_COLORS)
        self.output.pack(fill="both", expand=True, pady=(8, 10))
        self.output.configure(state="disabled")
        self.footer = tk.StringVar(value="Runs locally on this computer. No account, upload, or network connection is used.")
        ttk.Label(root, textvariable=self.footer, padding=(24, 0, 24, 12)).pack(fill="x")
        self._build_menus()
        self.load_preset()
        self.poll_id=self.root.after(100, self.poll)

    def _build_menus(self):
        menu = tk.Menu(self.root)
        self.examples_menu = tk.Menu(menu, tearoff=False)
        for i, (name, _) in enumerate(EXAMPLES):
            self.examples_menu.add_command(label=name, command=lambda index=i: self.choose_example(index))
        menu.add_cascade(label="Examples", menu=self.examples_menu)
        self.actions_menu = tk.Menu(menu, tearoff=False)
        self.actions_menu.add_command(label="Check matrices", accelerator="Command-Return", command=self.check_from_menu)
        self.actions_menu.add_command(label="Run selected maximum search", accelerator="Command-Shift-Return", command=self.search_from_menu)
        self.actions_menu.add_command(label="Stop", command=self.cancel)
        self.actions_menu.add_separator()
        self.actions_menu.add_command(label="Load found matrices into checker", command=self.load_found)
        self.actions_menu.add_command(label="Save result…", command=self.save_result)
        self.actions_menu.add_separator()
        self.actions_menu.add_command(label="Close", command=self.close)
        self.actions_menu.add_separator()
        self.actions_menu.add_command(label="Compare sets", command=self.compare)
        self.actions_menu.add_command(label="Canonical form of A", command=self.canonical)
        self.actions_menu.add_command(label="Replay saved certificate…", command=self.replay_file)
        menu.add_cascade(label="Actions", menu=self.actions_menu)
        self.research_menu = tk.Menu(menu, tearoff=False)
        research_actions = (
            ("Rank counts", self.distribution),
            ("Factor analysis", self.factor_analysis),
            ("Known bounds", self.bounds),
            ("Table through 5 by 5", self.table),
            ("Load best construction", lambda: self.construct("verified")),
            ("Two-row construction", lambda: self.construct("two_row")),
            ("Vandermonde construction", lambda: self.construct("vandermonde")),
            ("Copy A into B", self.copy_a_to_b),
            ("Known small classes", self.known_classes),
            ("Enumerate parameter domain", self.enumerate_classes),
            ("Add current set to collection", self.add_current),
            ("Group collected families", self.classify),
            ("Copy explanation", self.copy_explanation),
            ("Set research parameters…", self.set_research_parameters),
            ("Set class parameters…", self.set_class_parameters),
            ("Use current input for research parameters", self.use_input_parameters),
            ("Use current input for class parameters", lambda: self.use_input_parameters(classes=True)),
            ("Greedy search", lambda: self.run_search_method(0)),
            ("Exhaustive search", lambda: self.run_search_method(1)),
        )
        for title, command in research_actions:
            self.research_menu.add_command(label=title, command=command)
        menu.add_cascade(label="Research", menu=self.research_menu)
        self.views_menu = tk.Menu(menu, tearoff=False)
        for index, title in enumerate(("Check", "Bounds / search", "Compare", "Classes", "Explain")):
            self.views_menu.add_command(label=title, command=lambda i=index: self.tabs.select(i))
        menu.add_cascade(label="View", menu=self.views_menu)
        self.methods_menu = tk.Menu(menu, tearoff=False)
        for index, title in enumerate(("Fibre", "Support bounds", "Normalized combinations", "Binary ambient", "Full coefficients")):
            self.methods_menu.add_command(label=title, command=lambda i=index: self.check_with_method(i))
        menu.add_cascade(label="Check method", menu=self.methods_menu)
        options = tk.Menu(menu, tearoff=False)
        options.add_checkbutton(label="Allow square transpose", variable=self.transpose, command=self.input_changed)
        options.add_checkbutton(label="Allow common Frobenius", variable=self.frobenius, command=self.input_changed)
        menu.add_cascade(label="Equivalence options", menu=options)
        self.root.configure(menu=menu)
        modifier = "Command" if sys.platform == "darwin" else "Control"
        self.actions_menu.entryconfigure(0, accelerator=f"{modifier}-Return")
        self.actions_menu.entryconfigure(1, accelerator=f"{modifier}-Shift-Return")
        self.root.bind(f"<{modifier}-Return>", lambda event: (self.check_from_menu(), "break")[1])
        self.root.bind(f"<{modifier}-Shift-Return>", lambda event: (self.search_from_menu(), "break")[1])
        self.tabs.enable_traversal()

    def update_menus(self):
        if not hasattr(self, "actions_menu"):
            return
        busy = self.task is not None
        for i in range(len(EXAMPLES)):
            self.examples_menu.entryconfigure(i, state="disabled" if busy else "normal")
        for i in (0, 1):
            self.actions_menu.entryconfigure(i, state="disabled" if busy else "normal")
        self.actions_menu.entryconfigure(2, state="normal" if busy else "disabled")
        for i in (9,10,11):
            self.actions_menu.entryconfigure(i, state="disabled" if busy else "normal")
        found = bool(self.result_problems)
        self.actions_menu.entryconfigure(4, state="normal" if found and not busy else "disabled")
        self.actions_menu.entryconfigure(5, state="normal" if self.bundle and not busy else "disabled")
        for entry in range(self.research_menu.index("end") + 1):
            needs_result = entry == 12
            enabled = not busy and (not needs_result or self.bundle is not None)
            self.research_menu.entryconfigure(entry, state="normal" if enabled else "disabled")
        for entry in range(self.methods_menu.index("end") + 1):
            self.methods_menu.entryconfigure(entry, state="disabled" if busy else "normal")

    def check_with_method(self, index):
        if self.task is None:
            self.method.current(index)
            self.check_from_menu()

    def run_search_method(self, index):
        if self.task is None:
            self.search_method.current(index)
            self.search_from_menu()

    def use_input_parameters(self, classes=False):
        try:
            p = self.current_problem()
            matrices = p["generators"]
            m = len(matrices[0]) if matrices else p["rows"]
            n = len(matrices[0][0]) if matrices else p["columns"]
            variables = ((self.class_q,self.class_m,self.class_n,self.class_k) if classes
                         else (self.search_q,self.rows,self.columns,self.size))
            for variable, value in zip(variables, (p["field"]["order"],m,n,len(matrices) if classes else "")):
                variable.set(str(value))
            self.tabs.select(3 if classes else 1)
        except (ValueError, KeyError) as exc:
            messagebox.showerror("Check the input", str(exc), parent=self.root)

    def _parameter_dialog(self, variables):
        from tkinter import simpledialog
        initial = ", ".join(variable.get() for variable in variables)
        answer = simpledialog.askstring(
            "Research parameters", "Field q, rows, columns, optional set size k",
            initialvalue=initial, parent=self.root,
        )
        if answer is None:
            return
        values = [part.strip() for part in answer.split(",")]
        if len(values) not in (3, 4) or any(not value.isdigit() for value in values[:3]) or (len(values) == 4 and values[3] and not values[3].isdigit()):
            messagebox.showerror("Check the parameters", "Enter q, rows, columns and optionally k, separated by commas.", parent=self.root)
            return
        values += [""] * (4 - len(values))
        for variable, value in zip(variables, values):
            variable.set(value)

    def set_research_parameters(self):
        self._parameter_dialog((self.search_q, self.rows, self.columns, self.size))
        self.tabs.select(1)

    def set_class_parameters(self):
        self._parameter_dialog((self.class_q, self.class_m, self.class_n, self.class_k))
        self.tabs.select(3)

    def choose_example(self, index):
        if self.task is None:
            self.tabs.select(0)
            self.preset.current(index)
            self.load_preset()

    def check_from_menu(self):
        if self.task is None:
            self.tabs.select(0)
            self.check()

    def search_from_menu(self):
        if self.task is None:
            self.tabs.select(1)
            self.search()

    def _check_tab(self, parent):
        ttk.Label(parent, text="Start with a demonstration", style="Section.TLabel").pack(anchor="w")
        self.preset = ttk.Combobox(parent, values=[x[0] for x in EXAMPLES], state="readonly", width=43)
        self.preset.current(0)
        self.preset.pack(fill="x", pady=(7, 5))
        self.preset.bind("<<ComboboxSelected>>", lambda event: self.load_preset())
        self.controls.append((self.preset, "readonly"))
        bar = ttk.Frame(parent)
        bar.pack(fill="x", pady=(0, 14))
        self._button(bar, "Load matrices from JSON…", self.import_json).pack(side="left")
        self._button(bar, "Add to class collection", self.add_current).pack(side="left", padx=6)
        row = ttk.Frame(parent)
        row.pack(fill="x")
        ttk.Label(row, text="Field order q").pack(side="left")
        self.q = tk.StringVar(value="2")
        self.q_entry = ttk.Entry(row, textvariable=self.q, width=8)
        self.q_entry.pack(side="left", padx=(8, 0))
        self.controls.append((self.q_entry, "normal"))
        ttk.Label(parent, text="Extension polynomial, optional (coefficients highest degree first)",
                  wraplength=450).pack(anchor="w", pady=(10, 4))
        self.poly = tk.StringVar()
        self.poly_entry = ttk.Entry(parent, textvariable=self.poly)
        self.poly_entry.pack(fill="x")
        self.controls.append((self.poly_entry, "normal"))
        methods = ttk.Frame(parent)
        methods.pack(fill="x", pady=(6,0))
        ttk.Label(methods, text="Method").pack(side="left")
        self.method = ttk.Combobox(methods, values=("Automatic / fibre", "Support bounds + combinations", "All normalized combinations", "Binary ambient oracle", "Full coefficient oracle"), state="readonly", width=30)
        self.method.current(0)
        self.method.pack(side="left", padx=8)
        self.controls.append((self.method,"readonly"))
        self.method.bind("<<ComboboxSelected>>", self.input_changed)
        ttk.Label(parent, text="For F4, use 1,1,1. Leave blank for the library’s default field model.",
                  wraplength=450).pack(anchor="w", pady=(4, 12))
        ttk.Label(parent, text="Generators", style="Section.TLabel").pack(anchor="w")
        ttk.Label(parent, text="One matrix per block. Separate rows with new lines and matrices with a blank line.",
                  wraplength=450).pack(anchor="w", pady=(4, 6))
        bar = ttk.Frame(parent)
        bar.pack(side="bottom", fill="x", pady=(12, 0))
        self.check_button = self._button(bar, "Check matrices", self.check)
        self.check_button.pack(side="left")
        self._button(bar, "Rank counts", self.distribution).pack(side="left", padx=5)
        self._button(bar, "Factor analysis", self.factor_analysis).pack(side="left")
        self.editor = scrolledtext.ScrolledText(parent, wrap="none", height=8, font=("Menlo", 12),
                                               padx=10, pady=10, **TEXT_COLORS)
        self.editor.pack(fill="both", expand=True)
        self.editor.bind("<<Modified>>", self.input_changed)
        self.q.trace_add("write", self.input_changed)
        self.poly.trace_add("write", self.input_changed)

    def _entry_row(self, parent, fields):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=5)
        for label, variable, width in fields:
            ttk.Label(row, text=label).pack(side="left", padx=(0,4))
            entry = ttk.Entry(row, textvariable=variable, width=width)
            entry.pack(side="left", padx=(0,10))
            self.controls.append((entry,"normal"))
            variable.trace_add("write", self.input_changed)
        return row

    def _search_tab(self, parent):
        ttk.Label(parent, text="Known bounds and new constructions", style="Section.TLabel").pack(anchor="w")
        ttk.Label(parent, text="Fix the field and matrix size. A verified example gives a lower bound; a mathematical exclusion gives an upper bound.",
                  wraplength=440).pack(anchor="w", pady=(5,10))
        self.search_q = tk.StringVar(value="2")
        self.rows, self.columns = tk.StringVar(value="5"), tk.StringVar(value="5")
        self.size = tk.StringVar(value="")
        self._entry_row(parent, (("Field q",self.search_q,4),("Rows",self.rows,4),("Columns",self.columns,4)))
        row = ttk.Frame(parent); row.pack(fill="x",pady=8)
        self._button(row,"Known bounds",self.bounds).pack(side="left")
        self._button(row,"Table through 5 × 5",self.table).pack(side="left",padx=5)
        self._button(row,"Load best construction",lambda:self.construct("verified")).pack(side="left")
        ttk.Separator(parent).pack(fill="x",pady=10)
        ttk.Label(parent,text="Build a mathematical construction",style="Section.TLabel").pack(anchor="w")
        self._entry_row(parent,(("Set size k (blank = largest)",self.size,5),))
        row = ttk.Frame(parent); row.pack(fill="x",pady=4)
        self._button(row,"Two-row family",lambda:self.construct("two_row")).pack(side="left")
        self._button(row,"Vandermonde",lambda:self.construct("vandermonde")).pack(side="left",padx=6)
        self.infinity = tk.BooleanVar(value=True)
        self._option(parent,"Include infinity in Vandermonde parameters",self.infinity)
        self.independent_right = tk.BooleanVar(value=False)
        self._option(parent,"Use independent right factors when the size permits",self.independent_right)
        ttk.Separator(parent).pack(fill="x",pady=10)
        ttk.Label(parent,text="Search for a larger good basis",style="Section.TLabel").pack(anchor="w")
        row=ttk.Frame(parent);row.pack(fill="x",pady=6)
        self.search_method=ttk.Combobox(row,values=("Greedy constructions","Exhaustive tiny search"),state="readonly",width=25)
        self.search_method.current(0);self.search_method.pack(side="left")
        self.controls.append((self.search_method,"readonly"))
        self._button(row,"Run search",self.search).pack(side="left",padx=8)
        self.fresh = tk.BooleanVar(value=False)
        self._option(parent,"Start fresh instead of using saved research examples",self.fresh)
        ttk.Label(parent,text="A completed construction search may leave an unresolved bound gap. Exhaustive searches are practical only in small cases. Stop is always available.",
                  wraplength=440).pack(anchor="w",pady=8)

    def _option(self,parent,label,variable):
        widget=ttk.Checkbutton(parent,text=label,variable=variable,command=self.input_changed)
        widget.pack(anchor="w",pady=3)
        self.controls.append((widget,"normal"))
        return widget

    def _compare_tab(self,parent):
        ttk.Label(parent,text="Compare two good matrix spaces",style="Section.TLabel").pack(anchor="w")
        ttk.Label(parent,text="Set A is the current input in Check. Enter set B below, using the same field and polynomial model. One row change and one column change must match every generator direction.",
                  wraplength=440).pack(anchor="w",pady=(5,8))
        row=ttk.Frame(parent);row.pack(fill="x",pady=4)
        self._button(row,"Copy A into B",self.copy_a_to_b).pack(side="left")
        self._button(row,"Load B from JSON…",self.import_b).pack(side="left",padx=6)
        self.transpose = tk.BooleanVar(value=False)
        self.frobenius = tk.BooleanVar(value=False)
        self._option(parent,"Allow transpose (square matrices only)",self.transpose)
        self._option(parent,"Allow a common field automorphism (separate semilinear action)",self.frobenius)
        bar=ttk.Frame(parent);bar.pack(side="bottom",fill="x",pady=(8,0))
        self.compare_button=self._button(bar,"Compare A and B",self.compare)
        self.compare_button.pack(side="left")
        self._button(bar,"Canonical form of A",self.canonical).pack(side="left",padx=6)
        self.editor_b=scrolledtext.ScrolledText(parent,wrap="none",height=8,font=("Menlo",12),padx=10,pady=10,**TEXT_COLORS)
        self.editor_b.pack(fill="both",expand=True,pady=(6,0))
        self.editor_b.bind("<<Modified>>",self.input_changed)
        self.b_field = None
        self.b_empty_shape = None

    def _classes_tab(self,parent):
        ttk.Label(parent,text="Known classes and your example collection",style="Section.TLabel").pack(anchor="w")
        self.class_q=tk.StringVar(value="2");self.class_m=tk.StringVar(value="2")
        self.class_n=tk.StringVar(value="3");self.class_k=tk.StringVar(value="3")
        self._entry_row(parent,(("q",self.class_q,4),("Rows",self.class_m,4),("Columns",self.class_n,4),("k",self.class_k,4)))
        self._button(parent,"Show complete known small classes",self.known_classes).pack(anchor="w",pady=(4,8))
        self._button(parent,"Enumerate all good sets at this size",self.enumerate_classes).pack(anchor="w",pady=(0,6))
        ttk.Label(parent,text="Uses the transpose and field-automorphism options in Compare. Known complete domains include two-row shapes over F2–F5 and all shapes for k ≤ 3.",wraplength=440).pack(anchor="w")
        ttk.Separator(parent).pack(fill="x",pady=10)
        row=ttk.Frame(parent);row.pack(fill="x",pady=4)
        self._button(row,"Add A",self.add_current).pack(side="left")
        self._button(row,"Add JSON…",self.add_file).pack(side="left",padx=6)
        self._button(row,"Remove selected",self.remove_collected).pack(side="left")
        bar=ttk.Frame(parent);bar.pack(side="bottom",fill="x",pady=8)
        self._button(bar,"Group collected families",self.classify).pack(side="left")
        self._button(bar,"Clear collection",self.clear_collection).pack(side="left",padx=6)
        ttk.Label(parent,text="Grouping a collection reports classes among those supplied examples. It does not claim that every possible set has been found.",wraplength=440).pack(side="bottom",anchor="w",pady=6)
        self.collection_list=tk.Listbox(parent,height=6,font=("Helvetica",12),**{k:v for k,v in TEXT_COLORS.items() if k not in ('insertbackground',)})
        self.collection_list.pack(fill="both",expand=True,pady=6)
        self.controls.append((self.collection_list,"normal"))

    def _talk_tab(self, parent):
        ttk.Label(parent, text="What to tell the group", style="Section.TLabel").pack(anchor="w", pady=(0, 8))
        ttk.Button(parent, text="Copy this explanation", command=lambda: self.copy(TALK)).pack(side="bottom", anchor="w", pady=(10, 0))
        text = scrolledtext.ScrolledText(parent, wrap="word", font=("Helvetica", 13), padx=8, pady=8, **TEXT_COLORS)
        text.pack(fill="both", expand=True)
        text.insert("1.0", TALK)
        text.configure(state="disabled")

    def _button(self, parent, title, command):
        button = ttk.Button(parent, text=title, command=command)
        self.controls.append((button, "normal"))
        return button

    def clear_result(self):
        if not hasattr(self, "output"):
            return
        self.bundle, self.saved_text = None, ""
        self.result_problems = []
        self.result_picker.configure(values=(),state="disabled")
        self.status.configure(text="READY — run the checker", background="#e9edf5", foreground="#34435b")
        self.set_output("Choose Check, Bounds / search, Compare, or Classes.\n\nCheck validates one supplied set. Bounds and search concern its maximum possible size. Compare and Classes investigate allowed coordinate changes.\n\nSaved results include their complete evidence.")
        for button in (self.save_button, self.copy_button, self.found_button):
            button.configure(state="disabled")
        self.update_menus()

    def input_changed(self, *args):
        if args and isinstance(args[0], tk.Event) and isinstance(args[0].widget,tk.Text):
            widget=args[0].widget
            if not widget.edit_modified():
                return
            widget.edit_modified(False)
        if not self.loading and self.task is None:
            self.clear_result()

    def set_output(self, text):
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)
        self.output.configure(state="disabled")

    def load_problem(self, problem):
        self.loading = True
        try:
            self.q.set(str(problem["field"]["order"]))
            poly = problem["field"].get("irreducible_poly")
            self.poly.set(",".join(map(str, poly)) if poly else "")
            self.empty_shape = (problem.get("rows"),problem.get("columns")) if not problem["generators"] else None
            self.editor.delete("1.0", "end")
            self.editor.insert("1.0", matrices_text(problem["generators"]) if problem["generators"] else "[]")
            self.editor.edit_modified(False)
        finally:
            self.loading = False
        self.clear_result()

    def load_preset(self):
        try:
            self.load_problem(load_example(self.preset.current()))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            messagebox.showerror("Could not load example", str(exc), parent=self.root)

    def import_json(self):
        path = filedialog.askopenfilename(title="Load a matrix problem", filetypes=[("JSON files", "*.json")])
        if not path:
            return
        try:
            with open(path) as handle:
                data = json.load(handle)
            self.load_problem(unwrap_problem(data))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            messagebox.showerror("Could not load matrices", str(exc), parent=self.root)

    def check(self):
        try:
            problem = self.current_problem()
        except ValueError as exc:
            messagebox.showerror("Check the input", str(exc), parent=self.root)
            return
        method=("fibre","hybrid","normalized","binary_ambient","reference")[self.method.current()]
        self.start({"operation": "check", "problem": problem,"method":method})

    def search(self):
        try:
            request=self.parameter_request("search")
        except ValueError:
            messagebox.showerror("Check the dimensions", "Enter whole numbers for rows and columns.", parent=self.root)
            return
        request.update(method="greedy" if self.search_method.current()==0 else "exhaustive",use_known=not self.fresh.get())
        self.start(request)

    def current_problem(self):
        problem=parse_problem(self.editor.get("1.0","end"),self.q.get(),self.poly.get())
        if not problem["generators"]:
            if not self.empty_shape or any(type(x) is not int or x<1 for x in self.empty_shape):
                raise ValueError("Load an empty-basis JSON with explicit rows and columns first.")
            problem.update(rows=self.empty_shape[0],columns=self.empty_shape[1])
        return problem

    def parameter_request(self,operation):
        request={"operation":operation,"q":int(self.search_q.get()),"rows":int(self.rows.get()),"columns":int(self.columns.get())}
        if self.size.get().strip(): request['k']=int(self.size.get())
        return request

    def run_parameters(self,operation,**options):
        try:
            request=self.parameter_request(operation)
        except ValueError:
            messagebox.showerror("Check the parameters","Use whole numbers for q, rows, columns and optional set size.",parent=self.root)
            return
        self.start({**request,**options})

    def bounds(self): self.run_parameters("bounds")
    def table(self): self.run_parameters("table")
    def construct(self,kind):
        self.run_parameters("construct",construction=kind,include_infinity=self.infinity.get(),
                            variant="independent" if self.independent_right.get() else "circuit")

    def _run_current(self,operation,**options):
        try: problem=self.current_problem()
        except ValueError as exc:
            messagebox.showerror("Check set A",str(exc),parent=self.root);return
        self.start({"operation":operation,"problem":problem,**options})

    def distribution(self): self._run_current("distribution")
    def factor_analysis(self): self._run_current("factor_analysis")
    def canonical(self): self._run_current("canonical",allow_transpose=self.transpose.get(),allow_frobenius=self.frobenius.get())

    def _set_b(self,problem):
        self.loading=True
        try:
            self.b_field=problem['field']
            self.b_empty_shape=(problem.get('rows'),problem.get('columns'))
            self.editor_b.delete('1.0','end')
            self.editor_b.insert('1.0',matrices_text(problem['generators']) if problem['generators'] else '[]')
            self.editor_b.edit_modified(False)
        finally:self.loading=False
        self.clear_result()

    def copy_a_to_b(self):
        try:self._set_b(self.current_problem())
        except ValueError as exc:messagebox.showerror("Check set A",str(exc),parent=self.root)

    def import_b(self):
        path=filedialog.askopenfilename(title="Load set B",filetypes=[("JSON files","*.json")])
        if not path:return
        try:self._set_b(unwrap_problem(json.loads(open(path).read())))
        except (OSError,ValueError,KeyError,TypeError) as exc:messagebox.showerror("Could not load B",str(exc),parent=self.root)

    def compare(self):
        if self.task is not None:return
        try:
            a=self.current_problem()
            b=parse_problem(self.editor_b.get('1.0','end'),self.q.get(),self.poly.get())
            if self.b_field is not None:b['field']=self.b_field
            if not b['generators']:
                if not self.b_empty_shape:raise ValueError("Load B's explicit empty-basis shape.")
                b.update(rows=self.b_empty_shape[0],columns=self.b_empty_shape[1])
        except ValueError as exc:
            messagebox.showerror("Check the comparison inputs",str(exc),parent=self.root);return
        self.start({'operation':'equivalence','problem_a':a,'problem_b':b,
                    'allow_transpose':self.transpose.get(),'allow_frobenius':self.frobenius.get()})

    def known_classes(self):
        try:
            request={'operation':'known_classes','q':int(self.class_q.get()),'rows':int(self.class_m.get()),
                     'columns':int(self.class_n.get()),'k':int(self.class_k.get()),
                     'allow_transpose':self.transpose.get(),'allow_frobenius':self.frobenius.get()}
        except ValueError:
            messagebox.showerror("Check the parameters","Use whole numbers for q, rows, columns and k.",parent=self.root);return
        self.start(request)

    def enumerate_classes(self):
        try:
            request={'operation':'enumerate_classes','q':int(self.class_q.get()),'rows':int(self.class_m.get()),
                     'columns':int(self.class_n.get()),'k':int(self.class_k.get()),
                     'allow_transpose':self.transpose.get(),'allow_frobenius':self.frobenius.get()}
        except ValueError:
            messagebox.showerror("Check the parameters","Use whole numbers for q, rows, columns and k.",parent=self.root);return
        self.start(request)

    def _add_collected(self,problem):
        a=problem['generators'];m=len(a[0]) if a else problem.get('rows');n=len(a[0][0]) if a else problem.get('columns')
        self.collection.append(problem)
        self.collection_list.insert('end',f"{len(self.collection)}. F{problem['field']['order']} · {m} × {n} · {len(a)} generators")
        self.clear_result()
        self.footer.set(f"Collection contains {len(self.collection)} families. Open Classes to group them.")

    def add_current(self):
        try:self._add_collected(self.current_problem())
        except (ValueError,KeyError) as exc:messagebox.showerror("Check set A",str(exc),parent=self.root)

    def add_file(self):
        paths=filedialog.askopenfilenames(title="Add families to the collection",filetypes=[("JSON files","*.json")])
        try:
            for path in paths:self._add_collected(unwrap_problem(json.loads(open(path).read())))
        except (OSError,ValueError,KeyError,TypeError) as exc:messagebox.showerror("Could not add family",str(exc),parent=self.root)

    def remove_collected(self):
        for i in reversed(self.collection_list.curselection()):
            self.collection.pop(i);self.collection_list.delete(i)
        self.clear_result()

    def clear_collection(self):
        self.collection.clear();self.collection_list.delete(0,'end');self.clear_result()

    def classify(self):
        if not self.collection:
            messagebox.showinfo("Add examples first","Use Add A or Add JSON to collect matrix families.",parent=self.root);return
        self.start({'operation':'classify','problems':self.collection,
                    'allow_transpose':self.transpose.get(),'allow_frobenius':self.frobenius.get()})

    def replay_file(self):
        if self.task is not None:return
        path=filedialog.askopenfilename(title="Replay a saved certificate",filetypes=[("JSON files","*.json")])
        if not path:return
        try:data=json.loads(open(path).read())
        except (OSError,ValueError) as exc:
            messagebox.showerror("Could not read certificate",str(exc),parent=self.root);return
        self.start({'operation':'replay','data':data})

    def start(self, request):
        if self.task is not None:
            return
        try:
            seconds=float(self.work_seconds.get())
            if not 0<=seconds<=180:raise ValueError
        except ValueError:
            messagebox.showerror("Check work limit","Choose between 0 and 180 seconds.",parent=self.root);return
        request={**request,'max_seconds':seconds}
        self.clear_result()
        self.status.configure(text="RUNNING…", background="#e9edf5")
        self.set_output("Running exact finite-field checks.\n\nYou can stop this job; an interrupted computation will not be labelled a pass.")
        self.editor.configure(state="disabled")
        self.editor_b.configure(state="disabled")
        for control, _ in self.controls:
            control.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.job_id += 1
        self.task = ProcessTask(request, self.messages, self.job_id,timeout=seconds+10)
        threading.Thread(target=self.task.run, daemon=True).start()
        self.update_menus()
        self.footer.set(f"Computing locally. Work budget {seconds:g} seconds, with 10 seconds for setup and replay.")

    def cancel(self):
        if self.task is not None:
            self.task.cancel()
            self.status.configure(text="STOPPING…")

    def poll(self):
        try:
            while True:
                job_id, bundle = self.messages.get_nowait()
                if job_id != self.job_id:
                    continue
                self.task = None
                self.editor.configure(state="normal")
                self.editor_b.configure(state="normal")
                self.editor.edit_modified(False)
                self.editor_b.edit_modified(False)
                for control, state in self.controls:
                    control.configure(state=state)
                self.cancel_button.configure(state="disabled")
                self.display(bundle)
        except queue.Empty:
            pass
        self.poll_id=self.root.after(100, self.poll)

    def display(self, bundle):
        self.bundle = bundle
        status = bundle["status"]
        titles = {"VALID": "PASS", "INVALID": "FAIL", "EXACT": "EXACT MAXIMUM", "BOUNDED":"BOUNDS — GAP REMAINS",
                  "EQUIVALENT":"EQUIVALENT", "INEQUIVALENT":"INEQUIVALENT", "VERIFIED":"CERTIFICATE VERIFIED",
                  "COMPLETE_CLASSIFICATION":"COMPLETE IN STATED DOMAIN", "PARTIAL_CLASSIFICATION":"CLASSES IN SUPPLIED SCOPE",
                  "COMPLETE":"COMPLETE", "REJECTED":"CERTIFICATE REJECTED", "INVALID_INPUT":"CHECK INPUT FAMILIES",
                  "INCONCLUSIVE": "UNFINISHED", "ERROR": "INPUT / COMPUTATION ERROR", "CANCELLED": "STOPPED"}
        color = "#dff3e8" if status in ("VALID", "EXACT","EQUIVALENT","VERIFIED","COMPLETE","COMPLETE_CLASSIFICATION") else "#fde7e7" if status in ("INVALID", "ERROR","REJECTED","INVALID_INPUT") else "#fff0cf"
        self.status.configure(text=titles.get(status, status), background=color, foreground="#263641")
        self.saved_text = ("The job was stopped. No passing result is asserted." if status == "CANCELLED"
                           else explanation(bundle))
        self.set_output(self.saved_text)
        self.copy_button.configure(state="normal")
        self.save_button.configure(state="normal")
        self.result_problems=available_problems(bundle)
        if self.result_problems:
            self.found_button.configure(state="normal",text="Load selected matrices into checker")
            self.result_picker.configure(values=[label for label,_ in self.result_problems],state="readonly")
            self.result_picker.current(0)
        self.footer.set("Done. You can copy the explanation or save the complete result as JSON.")
        self.update_menus()

    def copy(self, text):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.footer.set("Explanation copied to the clipboard.")

    def copy_explanation(self):
        self.copy(self.saved_text)

    def save_result(self):
        if self.bundle is None:
            return
        path = filedialog.asksaveasfilename(title="Save result", defaultextension=".json", initialfile="checker-demo-result.json",
                                          filetypes=[("JSON files", "*.json")])
        if path:
            try:
                with open(path, "w") as handle:
                    json.dump(self.bundle, handle, indent=2, allow_nan=False)
                    handle.write("\n")
                self.footer.set("Result saved.")
            except OSError as exc:
                messagebox.showerror("Could not save result", str(exc), parent=self.root)

    def load_found(self):
        if self.result_problems:
            index=max(0,self.result_picker.current())
            problem=self.result_problems[index][1]
            self.tabs.select(0)
            self.load_problem(problem)

    def close(self):
        if self.task is not None:
            self.task.cancel()
        if getattr(self,'poll_id',None) is not None:
            self.root.after_cancel(self.poll_id)
            self.poll_id=None
        self.root.destroy()


def main():
    root = tk.Tk()
    DemoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
