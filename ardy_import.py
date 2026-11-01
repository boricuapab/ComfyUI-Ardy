import sys
from pathlib import Path


ARDY_ROOT = (
    Path(__file__).parent
    /
    "vendor"
    / "ardy"
)


def setup_ardy_path():

    path = str(ARDY_ROOT)

    if ARDY_ROOT.is_dir() and path not in sys.path:

        sys.path.insert(
            0,
            path
        )
