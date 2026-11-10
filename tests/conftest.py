"""Keep hosted macOS tests independent of an interactive Aqua session."""
import os
import sys

import pytest


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
