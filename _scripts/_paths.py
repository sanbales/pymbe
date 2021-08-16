from pathlib import Path
from shutil import which

import sys


ROOT = (Path(__file__) / "..").resolve()

DOWNLOADS = ROOT / "downloads"

ENV_ROOT = Path(sys.executable).parent

DOT_LOCATION = which("dot")
if DOT_LOCATION:
    DOT_LOCATION = Path(DOT_LOCATION)

POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "mysecretpassword"
POSTGRES_DB = "sysml2"

PSQL_DATA = ROOT / "data"
PSQL_LOGS = PSQL_DATA / "logs"
