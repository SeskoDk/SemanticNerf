import subprocess
from typing import Sequence


def run_command(cmd: Sequence[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)
