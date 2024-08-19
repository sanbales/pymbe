from importlib import resources as lib_resources
from pathlib import Path
from typing import Dict

import pytest

import pymbe
import pymbe.api as pm

PYMBE_ROOT = Path(pymbe.__file__).parent
TESTS_ROOT = Path(__file__).parent
FIXTURES = TESTS_ROOT / "fixtures"


def get_model_from_payload_form(filename: str) -> pm.Model:
    if not filename.endswith(".json"):
        filename += ".json"

    json_file = FIXTURES / filename

    return pm.Model.load_from_post_file(json_file)


@pytest.fixture
def load_kerml_library() -> pm.Model:
    """
    Load a digest of the KerML Library. Expectation is that the model will load with a series of Namespaces
    representing the specific libraries within the KerML library.
    """

    library_model = None

    with lib_resources.path("pymbe.static_data", "KernelLibrary.json") as lib_data:
        library_model = pm.Model.load_from_post_file(lib_data)

    return library_model
