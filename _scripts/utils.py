import subprocess
from pathlib import Path
from typing import List, Tuple, Union

import requests

from . import _paths as P
from . import _variables as V

HAS_COLOR = True

if V.WIN:
    HAS_COLOR = False
    try:
        import colorama

        colorama.init()
        HAS_COLOR = True
    except ImportError:
        print(
            f"Please install colorama in `{P.ENV_ROOT.name}` if you'd like pretty colors",
            flush=True,
        )


class COLOR:
    """Terminal colors. Always print ENDC when done :P

    Falls back to ASCII art in absence of colors (windows, no colorama)
    """

    HEADER = "\033[95m" if HAS_COLOR else "=== "
    OKBLUE = "\033[94m" if HAS_COLOR else "+++ "
    OKGREEN = "\033[92m" if HAS_COLOR else "*** "
    WARNING = "\033[93m" if HAS_COLOR else "!!! "
    FAIL = "\033[91m" if HAS_COLOR else "XXX "
    ENDC = "\033[0m" if HAS_COLOR else ""
    BOLD = "\033[1m" if HAS_COLOR else "+++ "
    UNDERLINE = "\033[4m" if HAS_COLOR else "___ "


def download_file(url: str, filename: Union[Path, str] = None, overwrite: bool = True) -> int:
    """Download a file from a URL."""
    try:
        if not filename:
            filename = url.split("?")[0].split("/")[-1]
        if isinstance(filename, str):
            if Path(filename).name == filename:
                filename = P.DOWNLOADS / filename
            else:
                filename = Path(filename)
            filename = filename.resolve()
        if filename.exists():
            if overwrite:
                print(f"{COLOR.WARNING} Will overwrite '{filename}' {COLOR.ENDC}")
        response = requests.get(url)
        if response.status_code == 200:
            if not filename.parent.exists():
                filename.parent.mkdir(parents=True)
            if filename.exists():
                filename.unlink()
            filename.write_bytes(response.content)
            print(f"{COLOR.OKGREEN} Downloaded {filename} from {url} {COLOR.ENDC}")
            return 0
    finally:
        pass
    print(f"{COLOR.WARNING} Failed to download: {url} {COLOR.ENDC}")
    return 1


def _run(args: Union[List[str], Tuple[str]], *, wait: bool = True, shorten_paths=False, **kwargs) -> int:
    blue, endc = COLOR.OKBLUE, COLOR.ENDC
    cwd = kwargs.get("cwd", None)
    if cwd:
        kwargs["cwd"] = str(cwd)
        location = f" in {cwd}"
    else:
        location = ""

    if kwargs.get("shell"):
        str_args = " ".join(map(str, args))
        print(
            f"{blue}\n==={location}\n{str_args}\n===\n{endc}",
            # .replace(str(P.ROOT), "."),
            flush=True,
        )
    else:
        str_args = list(map(str, args))
        print(
            f"{blue}\n===\n{' '.join(str_args)}{location}\n===\n{endc}",
            # .replace(str(P.ROOT), "."),
            flush=True,
        )
    proc = subprocess.Popen(str_args, **kwargs)

    if not wait:
        return proc

    result_code = 1
    try:
        result_code = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        result_code = proc.wait()

    return result_code


def _check_output(args, **kwargs):
    """wrapper for subprocess.check_output that handles non-string args"""
    return subprocess.check_output([*map(str, args)], **kwargs)
