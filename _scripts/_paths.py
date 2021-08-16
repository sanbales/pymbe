import sys
from pathlib import Path
from shutil import which

ROOT = (Path(__file__) / "../..").resolve()

DOWNLOADS = ROOT / "downloads"

ENV_ROOT = Path(sys.executable).parent

DOT_LOCATION = which("dot")
if DOT_LOCATION:
    DOT_LOCATION = Path(DOT_LOCATION)

POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "mysecretpassword"
POSTGRES_DB = "sysml2"

API = ROOT / "api"
PSQL_LOGS = API / "logs"
