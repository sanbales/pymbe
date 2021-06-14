""" Important Project Config Stuff. """

import os
from pathlib import Path
import platform

BASE_ENV = "base-pymbe"
BASE_ENV_FILE = "env-base.yml"

# platform
PLATFORM = os.environ.get("FAKE_PLATFORM", platform.system())
WIN = PLATFORM == "Windows"
OSX = PLATFORM == "Darwin"
LINUX = PLATFORM == "Linux"
UNIX = not WIN

# this directory (_scripts)
SCRIPTS = Path(__file__).parent

# the project root
ROOT = SCRIPTS.parent

# where the envs file are kept
ENVS_FILE_DIR = ROOT / "deploy" / "env_files"

# where to build the envs
ENVS_DIR = ROOT / "envs"

# lock file config
LOCK_DIR = ROOT / "deploy" / "lock_files"
