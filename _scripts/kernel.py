import json
import sys
import traceback
from argparse import ArgumentParser
from warnings import warn
from zipfile import ZipFile

from . import _paths as P
from . import _variables as V
from .utils import COLOR as C
from .utils import _check_output, _run, download_file

SYSML2_RELEASE_FOLDER = P.DOWNLOADS / f"SysML-v2-Release-{V.SYSML2_RELEASE}"

SYSML2_RELEASE_URL = f"{V.SYSML_RELEASE_GITHUB}/archive/refs/tags/{V.SYSML2_RELEASE}.zip"
SYSML2_RELEASE_ZIPFILE = P.DOWNLOADS / "sysml2_release.zip"
SYSML2_JUPYTER_FOLDER = SYSML2_RELEASE_FOLDER / "install/jupyter"
SYSML2_KERNEL_INSTALLER = SYSML2_JUPYTER_FOLDER / "jupyter-sysml-kernel/install.py"
SYSML2_LABEXTENSION = SYSML2_JUPYTER_FOLDER / "jupyterlab-sysml/package"

PARSER = ArgumentParser()

PARSER.add_argument(
    "--force-download",
    action="store_true",
    help="Force downloading the release",
)
PARSER.add_argument(
    "--skip-clean",
    action="store_true",
    help="Skip cleaning up downloaded files (e.g., .zip files)",
)
PARSER.add_argument(
    "--publish-to",
    default=V.REMOTE_API_SERVER_URL,
    type=str,
    help=f"IP address to publish SysML models to (default='{V.REMOTE_API_SERVER_URL}')",
)


def _get_sysml_release(force: bool = False, skip_clean: bool = False) -> list:
    exceptions = []
    # download and extract file
    if force or not SYSML2_RELEASE_ZIPFILE.exists():
        if (
            download_file(
                url=SYSML2_RELEASE_URL,
                filename=SYSML2_RELEASE_ZIPFILE,
            )
            != 0
        ):
            exceptions += [f"Failed to download file: {SYSML2_RELEASE_URL}"]
    # unzip the contents and remove the zip file
    try:
        with ZipFile(SYSML2_RELEASE_ZIPFILE) as zip_file:
            zip_file.extractall(path=P.DOWNLOADS)
        if not skip_clean:
            SYSML2_KERNEL_INSTALLER.unlink()
    except:  # pylint: disable=bare-except; # noqa: 722
        exceptions += [
            f"Could not unzip file '{SYSML2_RELEASE_ZIPFILE}' to '{P.DOWNLOADS}'",
            traceback.format_exc(),
        ]
    return exceptions


def install(
    force_download: bool = False,
    publish_to: str = V.REMOTE_API_SERVER_URL,
    skip_clean: bool = True,
) -> int:
    """Install SysML v2 Kernel for JupyterLab."""
    exceptions = []
    if not P.DOT_LOCATION:
        warn(f"Need to install graphviz in '{P.ENV_ROOT.name}' environment!")
        return 1

    if force_download or not SYSML2_KERNEL_INSTALLER.exists():
        exceptions += _get_sysml_release(
            force=force_download,
            skip_clean=skip_clean,
        )

    # remove old kernel
    try:
        output = _check_output(["jupyter", "kernelspec", "list", "--json"])
        kernel_specs = json.loads(output.decode())["kernelspecs"]
        found_old_kernel = "sysml" in kernel_specs
    except:  # pylint: disable=bare-except; # noqa: 722
        found_old_kernel = True
        exceptions += [f"Failed to get kernel specifications: {traceback.format_exc()}"]

    try:
        if found_old_kernel:
            if _run(["jupyter", "kernelspec", "remove", "sysml", "-f"], cwd=P.ROOT) != 0:
                exceptions += ["Failed to remove sysml kernel"]
    except:  # pylint: disable=bare-except; # noqa: 722
        exceptions += [f"Failed to remove sysml kernel: {traceback.format_exc()}"]

    # install sysmlv2 kernel
    if ":" not in publish_to:
        publish_to += f":{V.API_PORT}"

    kernel_install_commands = [
        "python",
        SYSML2_KERNEL_INSTALLER.resolve(),
        "--sys-prefix",
        f"--api-base-path={publish_to}",
        f"--graphviz-path={P.DOT_LOCATION.resolve().as_posix()}",
    ]
    if _run(kernel_install_commands, cwd=P.ROOT) != 0:
        exceptions += [
            f"Failed to install kernel: {SYSML2_RELEASE_URL}",
        ]

    if _run(["jupyter", "labextension", "uninstall", "jupyterlab-sysml"], cwd=P.ROOT) != 0:
        exceptions += ["Failed to uninstall the JupyterLab SysML extension"]

    if (
        _run(["jupyter", "labextension", "install", SYSML2_LABEXTENSION.as_posix()], cwd=P.ROOT)
        != 0
    ):
        exceptions += ["Failed to install the JupyterLab SysML extension"]

    if exceptions:
        print(f"{C.WARNING} Error installing SysML 2 kernel! {C.ENDC}")
        for msg in exceptions:
            print(f"{C.WARNING} {msg} {C.ENDC}")

    return 1 if exceptions else 0


if __name__ == "__main__":
    args, extra_args = PARSER.parse_known_args(sys.argv[1:])

    sys.exit(
        install(
            force_download=args.force_download,
            publish_to=args.publish_to,
            skip_clean=args.skip_clean,
        )
    )
