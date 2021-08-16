from argparse import ArgumentParser
from contextlib import closing
from time import sleep

import socket

from . import _paths as P
from . import _variables as V
from .utils import _run, COLOR as C


API_ZIP_FILE = P.DOWNLOADS / "api_server.zip"
API_DOWNLOAD_URL = f"{V.SYSML_API_GITHUB}/archive/refs/tags/{V.SYSML2_API_RELEASE}.zip"

SBT_ZIP_FILE = P.DOWNLOADS / "sbt.zip"
SBT_DOWNLOAD_URL = f"{V.SBT_GITHUB}/releases/download/v{V.SBT_VERSION}/sbt-{V.SBT_VERSION}.zip"

PARSER = ArgumentParser()

PARSER.add_argument("--download", action="store_true")
PARSER.add_argument(
    "--psql-port",
    default=5432,
    type=int,
    help="Port on which to run PosgreSQL (default=5432)",
)
PARSER.add_argument("--start", action="store_true")
PARSER.add_argument("--stop", action="store_true")

PSQL = "psql"


class PostgreSQLProcess:
    """A context manager for a psql process."""

    MAX_RETRIES = 10
    WAIT_BETWEEN_RETRIES = 0.5  # sec

    def __init__(self, host, port, autostart=True):
        self.host = host
        self.port = port
        self.__proc = None

        if not P.PSQL_LOGS.exists():
            P.PSQL_DATA.mkdir(parents=True, exist_ok=True)
            P.PSQL_LOGS.parent.mkdir(parents=True, exist_ok=True)
            P.PSQL_LOGS.touch(mode=0o777)

        if autostart:
            self.start_proc()

    def start_proc(self):
        if self.is_running():
            print(
                f"{C.WARNING} Mongo is already running "
                f"on {self.host}:{self.port} {C.ENDC}"
            )
            return

        self.__proc = _run(
            [
                PSQL,
                "--dbpath",
                MONGO_DATA.resolve(),
                "--no-password",
                f"--port={self.port}",
                f"--log-file={P.PSQL_LOGS.resolve()}",
            ],
            cwd=MONGO_DATA,
            wait=False,
        )
        retries = 0
        while not self.is_running() and retries < self.MAX_RETRIES:
            retries += 1
            sleep(self.WAIT_BETWEEN_RETRIES)

        if not self.is_running():
            self.__proc.terminate()
            self.__proc.wait()
            print(f"{C.WARNING} Could not start mongo {C.ENDC}")
        else:
            print(f"{C.OKGREEN} Serving mongo on {self.host}:{self.port} {C.ENDC}")

    def is_running(self):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            return sock.connect_ex((self.host, self.port)) == 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        print(f"{C.OKGREEN} Shutting down mongo on {self.host}:{self.port} {C.ENDC}")
        proc = self.__proc
        if proc is not None:
            self.__proc.terminate()
            self.__proc.wait()
