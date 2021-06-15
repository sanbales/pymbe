""" doit tasks for PyMBE """

import os

from ensureconda.resolve import platform_subdir

from scripts import project_config as P

os.environ.update(
    NODE_OPTS="--max-old-space-size=4096",
    PYTHONIOENCODING="utf-8",
    PIP_DISABLE_PIP_VERSION_CHECK="1",
    MAMBA_NO_BANNER="1",
)

DOIT_CONFIG = {
    "verbosity": 2,
    "default_tasks": ["build_envs", "dev_setup", "test",],
}


def _activate_cmd(env):
    if P.CI:
        return f""
    elif P.WIN:
        return f"activate {P.ENVS_DIR}\{env} &&"
    else:
        return f"source activate {P.ENVS_DIR}/{env} &&"


def task_build_envs():
    """
    Build environments from environment files in /deploy/envs. Skip BASE_ENV.
    """
    platform_key = platform_subdir()
    env_files = [
        child
        for child in P.LOCK_DIR.iterdir()
        if (platform_key in child.stem and child.suffix == ".lock")
    ]

    # make the envs dir if it doesn't exist
    P.ENVS_DIR.mkdir(parents=True, exist_ok=True)

    # make all envs there are lock files for, but validate the platform matches host first
    for file in env_files:

        env_name = file.stem.split("-")[0]
        yield dict(
            name=f"build_envs-{file.stem}",
            actions=[
                # f"conda-lock -p={P.ENVS_DIR}, --validate-platform --mamba {str(file)}"
                f"mamba create --file {str(file)} -p={str(P.ENVS_DIR / env_name)} -yq"
            ],
        )


def task_dev_setup():
    """Setup development environment"""
    return {
        "actions": [
            f"{_activate_cmd('developer')} git submodule update --init",
            f"{_activate_cmd('developer')} pip install git+https://github.com/Systems-Modeling/SysML-v2-API-Python-Client.git --no-dependencies",
            f"{_activate_cmd('developer')} pip install -e . --no-dependencies",
        ]
    }


def task_lab():
    """Launch JupyterLab"""
    if P.WIN:
        return {"actions": [f"{_activate_cmd('developer')} jupyter lab"]}
    else:
        return {
            "actions": [
                f"{_activate_cmd('developer')} jupyter lab --no-browser --debug"
            ]
        }


def task_lint():
    """Apply linting to the codebase"""
    if P.WIN:
        return {"actions": [f"{_activate_cmd('developer')} isort . && black ."]}
    else:
        return {"actions": [f"{_activate_cmd('developer')} isort . && black src/"]}


def task_package():
    """Make a source distribution"""
    return {"actions": [f"{_activate_cmd('developer')} python setup.py sdist"]}


def task_test():
    """Run unit tests."""
    return {"actions": [f"{_activate_cmd('developer')} py.test tests/"]}


def task_tutorial():
    """Launch JupyterLab and open Tutorial Notebook"""
    if P.WIN:
        return {
            "actions": [
                f"{_activate_cmd('developer')} jupyter lab notebooks/Tutorial.ipynb"
            ]
        }
    else:
        return {
            "actions": [
                f"{_activate_cmd('developer')} jupyter lab --no-browser --debug notebooks/Tutorial.ipynb"
            ]
        }


def task_update():
    """
    Update lock files to reflect environment files in /deploy/envs. Skip BASE_ENV.
    """
    env_files = [
        child
        for child in P.ENVS_FILE_DIR.iterdir()
        if P.BASE_ENV_FILE not in str(child)
    ]

    for file in env_files:
        lock_file_template = f'"{file.stem}-{{platform}}.lock"'
        yield dict(
            name=f"update-{file.stem}",
            actions=[
                f"conda-lock -f {str(file)} --filename-template {lock_file_template}"
            ],
        )

    def _move_lock_files():
        # move lock files to convenient location
        lock_files = [child for child in P.ROOT.iterdir() if child.suffix == ".lock"]
        P.LOCK_DIR.mkdir(exist_ok=True, parents=True)
        for lock_file in lock_files:
            lock_file.replace(str(P.LOCK_DIR / lock_file.name))

    yield {
        "name": "update-move-locks",
        "actions": [_move_lock_files],
    }
