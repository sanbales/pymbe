# pylint: disable=redefined-outer-name
from pathlib import Path
from typing import Dict, List

import pytest

import pymbe
import pymbe.api as pm
from pymbe.graph import SysML2LabeledPropertyGraph

PYMBE_ROOT = Path(pymbe.__file__).parent
TESTS_ROOT = Path(__file__).parent
FIXTURES = TESTS_ROOT / "fixtures"


def get_model_from_payload_form(filename: str) -> pm.Model:
    if not filename.endswith(".json"):
        filename += ".json"

    json_file = FIXTURES / filename

    return pm.Model.load_from_post_file(json_file)


def get_model(filename: str) -> pm.Model:
    if not filename.endswith(".json"):
        filename += ".json"

    json_file = FIXTURES / filename
    if not json_file.exists():
        fixtures_exists = FIXTURES.exists()
        contents = ""
        if fixtures_exists:
            contents = "\nContents:\n" + "\n".join(map(str, FIXTURES.glob("*")))
        raise ValueError(
            f"Could not load: '{json_file.absolute()}'!\n"
            "Did you forget to run git submodules? If so, run:\n"
            "  git submodule update --init\n"
            f"pyMBE is here: {PYMBE_ROOT.absolute()}\n"
            f"{TESTS_ROOT} exists: {TESTS_ROOT.exists()}\n"
            f"{FIXTURES} exists: {fixtures_exists}" + contents
        )

    return pm.Model.load_from_file(json_file)


@pytest.fixture
def basic_load_files() -> Dict[str, pm.Model]:
    level1 = get_model_from_payload_form("Model_Loader_Test_Level_1")
    level2 = get_model_from_payload_form("Model_Loader_Test_Level_2")
    level3 = get_model_from_payload_form("Model_Loader_Test_Level_3")
    literal = get_model_from_payload_form("Model_Loader_Literal_Test")
    annex_a = get_model_from_payload_form("Annex_A_Bike_Example")
    return {
        "Level1": level1,
        "Level2": level2,
        "Level3": level3,
        "Literals": literal,
        "AnnexA": annex_a,
    }
