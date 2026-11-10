"""Match desktop test lifetimes to a normal single-app launch."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def isolated_windows_gui(request):
    """Avoid retaining Tcl interpreter state between Windows app launches.

    The assertions still execute in a real Tk window; each selected GUI test
    gets a fresh Python process. The opt-in variable exercises this harness on
    a local Mac too. A failed child remains a failed parent test.
    """
    isolate = sys.platform == 'win32' or os.environ.get('RANKONE_ISOLATE_GUI_TESTS') == '1'
    if not isolate or os.environ.get('RANKONE_GUI_TEST_CHILD') == '1':
        return False
    env = dict(os.environ, RANKONE_GUI_TEST_CHILD='1')
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', request.node.nodeid],
        cwd=Path(__file__).resolve().parents[1], env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    return True


def pytest_collection_modifyitems(items):
    if sys.platform != 'darwin' or os.environ.get('GITHUB_ACTIONS') != 'true':
        return
    desktop_names = {
        'test_check_button_stays_inside_visible_window_at_supported_sizes',
        'test_all_three_text_panels_have_readable_contrast',
        'test_all_tabs_have_accessible_controls_at_supported_window_sizes',
        'test_menu_routes_and_current_input_parameters',
    }
    for item in items:
        name = item.name.split('[', 1)[0]
        if name.startswith('test_native_') or name in desktop_names:
            item.add_marker(pytest.mark.skip(reason=(
                'Interactive Tk desktop required; desktop tests run on local '
                'macOS and Windows, not the hosted macOS runner.'
            )))
