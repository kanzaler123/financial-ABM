"""Work around a hung Windows WMI service (seen 2026-10-16, after the machine ran out of memory).

Python 3.12's platform.uname() queries WMI and then blocks forever; importing scipy.stats reaches it through
numpy.testing, so the grid summary and analyze.py hung. Pre-filling platform's uname cache from values that do
not need WMI avoids the query. Run identities use platform.python_version(), which parses sys.version and never
calls uname, so no identity or number changes.
"""

from __future__ import annotations

import os
import platform
import sys


def prefill_uname() -> None:
    if sys.platform != "win32" or platform._uname_cache is not None:
        return
    v = sys.getwindowsversion()
    platform._uname_cache = platform.uname_result(
        "Windows", os.environ.get("COMPUTERNAME", ""), "11" if v.build >= 22000 else "10",
        f"{v.major}.{v.minor}.{v.build}", os.environ.get("PROCESSOR_ARCHITECTURE", "AMD64"))
