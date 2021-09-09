import re
import sys
import traceback
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

from . import _paths as P
from . import _variables as V
from .utils import COLOR as C
from .utils import _check_output, _run, download_file

API_ZIP_FILE = P.DOWNLOADS / "api_server.zip"
API_DOWNLOAD_URL = f"{V.SYSML_API_GITHUB}/archive/refs/tags/{V.SYSML2_API_RELEASE}.zip"

SBT_ZIP_FILE = P.DOWNLOADS / "sbt.zip"
SBT_DOWNLOAD_URL = f"{V.SBT_GITHUB}/releases/download/v{V.SBT_VERSION}/sbt-{V.SBT_VERSION}.zip"

# Executables
PG_CTL = "pg_ctl"
PSQL = "psql"
SBT = P.SBT_BIN / ("sbt" + (".bat" if V.WIN else ""))

PARSER = ArgumentParser()

PARSER.add_argument(
    "--host",
    default="127.0.0.1",
    type=str,
    help="Hostname on which to run PosgreSQL (default='127.0.0.1')",
)
PARSER.add_argument(
    "--port",
    default=5432,
    type=int,
    help="Port on which to run PosgreSQL (default=5432)",
)
PARSER.add_argument("--setup", action="store_true")
PARSER.add_argument("--silent", action="store_true")
PARSER.add_argument("--start", action="store_true")
PARSER.add_argument("--stop", action="store_true")
PARSER.add_argument("--tear-down", action="store_true")
PARSER.add_argument("--no-autostart", action="store_true")


# Commands:
# 1. Create User `postgres`
# 2. Create database 'sysml2'
# 3.


def setup_api(force=False, skip_clean=True):
    """Setup the SysML v2 Pilot API and the Scala Build Tools (SBT)"""
    if not P.API.exists() or not list(P.API.iterdir()):
        if force or not API_ZIP_FILE.exists():
            if (
                download_file(
                    url=API_DOWNLOAD_URL,
                    filename=API_ZIP_FILE,
                )
                != 0
            ):
                print(
                    f"{C.FAIL} Failed to download the SysML v2 Pilot API!"
                    f"  (url={API_DOWNLOAD_URL} {C.ENDC}"
                )
        # unzip the contents and remove the zip file
        try:
            with ZipFile(API_ZIP_FILE) as zip_file:
                zip_file.extractall(path=P.API)
            if not skip_clean:
                API_ZIP_FILE.unlink()
        except:  # pylint: disable=bare-except; # noqa: 722
            print(f"{C.FAIL} Could not unzip file '{API_ZIP_FILE}' to '{P.API}' {C.ENDC}")
            print(f"{C.FAIL} {traceback.format_exc()} {C.ENDC}")

    if not P.SBT_BIN.exists():
        # download and extract file
        if force or not SBT_ZIP_FILE.exists():
            if (
                download_file(
                    url=SBT_DOWNLOAD_URL,
                    filename=SBT_ZIP_FILE,
                )
                != 0
            ):
                print(f"{C.FAIL} Failed to download SBT!  (url={SBT_DOWNLOAD_URL} {C.ENDC}")
        # unzip the contents and remove the zip file
        try:
            with ZipFile(SBT_ZIP_FILE) as zip_file:
                zip_file.extractall(path=P.API)
            if not skip_clean:
                SBT_ZIP_FILE.unlink()
        except:  # pylint: disable=bare-except; # noqa: 722
            print(f"{C.FAIL} Could not unzip file '{SBT_ZIP_FILE}' to '{P.API}' {C.ENDC}")
            print(f"{C.FAIL} {traceback.format_exc()} {C.ENDC}")


def tear_down():
    try:
        P.PSQL_DATA.unlink(missing_ok=True)
    except:  # pylint: disable=bare-except; # noqa: 722
        print(f"{C.FAIL} Failed to delete the PostgreSQL data folder {C.ENDC}")
        print(f"{C.FAIL} {traceback.format_exc()} {C.ENDC}")
    try:
        P.API.unlink(missing_ok=True)
    except:  # pylint: disable=bare-except; # noqa: 722
        print(f"{C.FAIL} Failed to delete the API folder {C.ENDC}")
        print(f"{C.FAIL} {traceback.format_exc()} {C.ENDC}")


@dataclass
class PostgreSQLProcess:  # pylint: disable=too-many-instance-attributes
    """
    A context manager for a psql process.

    IMPORTANT: This configuration is not intended to be used in production!

    """

    pid_regex = re.compile(r"\(PID: ?(\d+)\)")

    host: str = "127.0.0.1"
    port: int = 5432
    autostart: bool = True
    silent: bool = False

    _pid: int = None

    _dbname: str = V.PSQL_DBNAME
    _datafile: Path = P.PSQL_DATA.resolve().absolute()
    _logsfile: Path = P.PSQL_LOGS.resolve().absolute()
    _pwfile: Path = P.PSQL_PWFILE.resolve().absolute()
    _username: str = V.PSQL_USERNAME

    def __post_init__(self):
        if "0.0.0.0" in self.host:
            print(
                f"{C.WARNING} You are binding to PostgreSQL to 0.0.0.0, "
                "this exposes serious risks! {C.ENDC}"
            )

        if not self._datafile.exists() or not list(self._datafile.iterdir()):
            self.initialize_db()

        if not self._logsfile.exists():
            self._logsfile.parent.mkdir(parents=True, exist_ok=True)

        if self.autostart:
            self.start_proc()

    def clear(self):
        self.server("stop")
        self._datafile.unlink()

    def write_pwfile(self):
        """
        Write password to pwfile

        IMPORTANT: This is NOT secure!!!  It should NOT be used in production!
        """
        if self._pwfile.exists():
            self._pwfile.unlink()
        self._pwfile.write_text(f"{V.PSQL_PASSWORD}\n")

    def initialize_db(self):
        P.PSQL_DATA.mkdir(parents=True, exist_ok=True)

        # Initialize the database
        silent_args = ["--silent"] if self.silent else []
        result_code = _run(
            [
                PG_CTL,
                "initdb",
                f"--pgdata={self._datafile}",
                f"--pwfile={self._pwfile}",
                f"--username={self._username}",
            ]
            + silent_args,
            cwd=P.ROOT,
            wait=True,
        )
        if result_code != 0:
            print(f"{C.FAIL} Database initialization (initdb) returned a non-zero value {C.ENDC}")

    def create_db(self):
        # Create the sysml database
        is_running = self.is_running()
        if not is_running:
            self.server("start")

        result_code = _run(
            [
                "createdb",
                f"--owner={self._username}",
                f"--host={self.host}",
                f"--port={self.port}",
                "--no-password",
                self._dbname,
            ],
            cwd=self._datafile,
            wait=True,
        )

        if result_code != 0:
            print(f"{C.FAIL} Database creation (createdb) returned a non-zero value {C.ENDC}")
        if not is_running:
            self.server("stop")

    def server(self, command: str, wait=False):
        func = _check_output if wait else _run
        print(f"{C.OKBLUE} running '{command}' server command {C.ENDC}")

        os_dep_args = ["-U", V.PSQL_USERNAME] if V.WIN else []
        return func(
            [
                PG_CTL,
                f"--pgdata={self._datafile}",
                f"--log={self._logsfile}",
                command,
            ]
            + os_dep_args,
            cwd=P.ROOT,
            wait=wait,
        )

    def server_status(self):
        return self.server("status", wait=True).decode()

    def start_proc(self):
        if self.is_running():
            print(f"{C.WARNING} PostgreSQL is already running {C.ENDC}")
            return

        self.server("start")

        if not self.is_running():
            print(f"{C.FAIL} Could not start PostgreSQL {C.ENDC}")
        else:
            print(f"{C.OKGREEN} Serving PostgreSQL on {self.host}:{self.port} {C.ENDC}")

    def is_running(self):
        pids = self.pid_regex.findall(self.server_status)
        self._pid = pid = int(pids[0]) if pids else None
        return pid

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        print(f"{C.OKBLUE} Shutting down psql on {self.host}:{self.port} {C.ENDC}")
        self.server("stop")
        print(f"{C.OKGREEN} Finished shutting down psql on {self.host}:{self.port} {C.ENDC}")


def setup():
    pass


if __name__ == "__main__":
    args, extra_args = PARSER.parse_known_args(sys.argv[1:])

    result_codes = []
    if args.tear_down:
        result_codes.append(tear_down())

    if args.setup:
        result_codes.append(setup())

    with PostgreSQLProcess(
        host=args.host,
        port=args.port,
        autostart=not args.no_autostart,
    ) as psql_proc:
        if args.start:
            result_codes.append(
                # launch(env=env, port=args.mbee_port)
            )

    sys.exit(max(result_codes))
