import subprocess
import sys
from pathlib import Path


def install():
    requirements = Path(__file__).parent / "requirements.txt"

    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        str(requirements)
    ])


if __name__ == "__main__":
    install()