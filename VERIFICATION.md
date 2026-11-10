# Release verification: 0.2.1

Verified locally on macOS with Python 3.12 on November 10, 2026.

- Clean public export: **328 tests passed in 48.95 seconds**. Tests run without the original private research directory.
- Wheel built successfully; field models and mathematical data are included as package resources.
- On-screen checks through the native menus: all five checking methods passed the F2 triple; the F3 triple displayed an explicit extra rank-one counterexample; F4 arithmetic passed its extension-field example.
- On-screen research operations: exact rank counts, paired factor analysis, exact binary maximum, unresolved F5 bound, table through 5 by 5, known small classes, complete F2 2-by-3 enumeration, loading a representative, canonical form, grouping supplied examples and equivalence transformations. Two-row and Vandermonde constructions and both search-method menu routes also returned the expected decisions. A saved search report was reopened through the normal file picker and displayed CERTIFICATE VERIFIED.
- The F4 comparison pair was observed as linearly inequivalent and equivalent when a common Frobenius automorphism was enabled.
- Automated desktop tests cover editable inputs, result invalidation, saved reports, replay, constructor/search wiring, controls within supported window sizes and clipboard content. The computer-control connection intermittently timed out on Tk dialogs; these individual dialog interactions are not claimed as a complete manual walkthrough.
- Windows launchers require Python 3.12, the Python launcher, and Tcl/Tk. Windows keyboard shortcuts and whole-job cancellation have regression coverage. The local Mac run does not establish Windows execution; consult the repository's **Actions** results for actual Windows, Linux and macOS runs.

Passing tests and finite computational certificates do not establish a general theorem or settle unresolved maximum/classification cases. Computation limits and classification scope are displayed in the app and stored in its reports.
