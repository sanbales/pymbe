from pathlib import Path
from typing import Dict

import pytest
import traitlets as trt

import pymbe
import pymbe.api as pm
from pymbe.client import SysML2Client
from pymbe.graph import SysML2LabeledPropertyGraph
from pymbe.interpretation.interp_playbooks import (
    build_expression_sequence_templates,
    build_sequence_templates,
    create_set_with_new_instances,
    random_generator_phase_1_multiplicities,
    random_generator_playbook_phase_1_singletons,
    random_generator_playbook_phase_2_rollup,
    random_generator_playbook_phase_2_unconnected,
    random_generator_playbook_phase_3,
    random_generator_playbook_phase_4,
    random_generator_playbook_phase_5,
)
from pymbe.local.stablization import build_stable_id_lookups

PYMBE_ROOT = Path(pymbe.__file__).parent
TESTS_ROOT = Path(__file__).parent
FIXTURES = TESTS_ROOT / "fixtures"


def get_client(filename: str) -> SysML2Client:
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

    helper_client = SysML2Client()
    helper_client._load_from_file(json_file)

    return helper_client


def kerbal_model_loaded_client() -> SysML2Client:
    return get_client("Kerbal")


def simple_parts_model_loaded_client() -> SysML2Client:
    return get_client("Simple Parts Model")


def simple_actions_model_loaded_client() -> SysML2Client:
    return get_client("Simple Actions Example")


@pytest.fixture
def kerbal_client() -> SysML2Client:
    return kerbal_model_loaded_client()


@pytest.fixture
def kerbal_ids_by_type(kerbal_client) -> dict:
    return {
        metatype: [element._id for element in elements]
        for metatype, elements in kerbal_client.model.ownedMetatype.items()
    }


@pytest.fixture
def kerbal_stable_names():
    client = kerbal_model_loaded_client()
    lpg = SysML2LabeledPropertyGraph()
    lpg.model = client.model
    return build_stable_id_lookups(lpg)


@pytest.fixture()
def all_kerbal_names(kerbal_client) -> list:
    all_elements = kerbal_client.model.elements

    return [element._data["name"] for element in all_elements.values() if "name" in element._data]


@pytest.fixture
def kerbal_lpg() -> SysML2LabeledPropertyGraph:
    new_lpg = SysML2LabeledPropertyGraph()
    client = kerbal_model_loaded_client()
    trt.link(
        (client, "model"),
        (new_lpg, "model"),
    )
    return new_lpg


@pytest.fixture
def kerbal_random_stage_1_instances(kerbal_lpg) -> dict:
    ptg = kerbal_lpg.get_projection("Part Typing")
    scg = kerbal_lpg.get_projection("Part Definition")

    full_multiplicities = random_generator_phase_1_multiplicities(kerbal_lpg, ptg, scg)

    return {
        type_id: create_set_with_new_instances(
            sequence_template=[kerbal_lpg.model.elements[type_id]],
            quantities=[number],
        )
        for type_id, number in full_multiplicities.items()
    }


@pytest.fixture
def kerbal_random_stage_1_complete(kerbal_lpg, kerbal_random_stage_1_instances) -> dict:
    scg = kerbal_lpg.get_projection("Part Definition")

    random_generator_playbook_phase_1_singletons(
        kerbal_lpg.model,
        scg,
        kerbal_random_stage_1_instances,
    )

    return kerbal_random_stage_1_instances


@pytest.fixture
def kerbal_random_stage_2_complete(kerbal_lpg, kerbal_random_stage_1_complete) -> dict:
    scg = kerbal_lpg.get_projection("Part Definition")

    random_generator_playbook_phase_2_rollup(scg, kerbal_random_stage_1_complete)

    random_generator_playbook_phase_2_unconnected(kerbal_lpg.model, kerbal_random_stage_1_complete)

    return kerbal_random_stage_1_complete


@pytest.fixture
def kerbal_random_stage_3_complete(kerbal_lpg, kerbal_random_stage_2_complete) -> dict:
    feature_sequences = build_sequence_templates(lpg=kerbal_lpg)

    random_generator_playbook_phase_3(
        kerbal_lpg.model,
        feature_sequences,
        kerbal_random_stage_2_complete,
    )

    return kerbal_random_stage_2_complete


@pytest.fixture
def kerbal_random_stage_4_complete(
    kerbal_lpg: SysML2LabeledPropertyGraph,
    kerbal_random_stage_3_complete: dict,
) -> dict:
    expr_sequences = build_expression_sequence_templates(lpg=kerbal_lpg)

    random_generator_playbook_phase_4(
        kerbal_lpg.model,
        expr_sequences,
        kerbal_random_stage_3_complete,
    )

    return kerbal_random_stage_3_complete


@pytest.fixture
def kerbal_random_stage_5_complete(
    kerbal_lpg: SysML2LabeledPropertyGraph,
    kerbal_random_stage_4_complete: dict,
) -> dict:

    random_generator_playbook_phase_5(
        kerbal_lpg, kerbal_lpg.get_projection("Connection"), kerbal_random_stage_4_complete
    )

    return kerbal_random_stage_4_complete


@pytest.fixture
def simple_parts_client() -> SysML2Client:
    return simple_parts_model_loaded_client()


@pytest.fixture
def simple_parts_lpg() -> SysML2LabeledPropertyGraph:
    new_lpg = SysML2LabeledPropertyGraph()
    client = simple_parts_model_loaded_client()

    new_lpg.model = client.model

    return new_lpg


@pytest.fixture
def simple_parts_stable_names():
    client = simple_parts_model_loaded_client()
    lpg = SysML2LabeledPropertyGraph()
    lpg.model = client.model
    return build_stable_id_lookups(lpg)


@pytest.fixture
def simple_parts_random_stage_1_instances(simple_parts_lpg) -> dict:
    ptg = simple_parts_lpg.get_projection("Part Typing")
    scg = simple_parts_lpg.get_projection("Part Definition")

    full_multiplicities = random_generator_phase_1_multiplicities(simple_parts_lpg, ptg, scg)

    return {
        type_id: create_set_with_new_instances(
            sequence_template=[simple_parts_lpg.model.elements[type_id]],
            quantities=[number],
        )
        for type_id, number in full_multiplicities.items()
    }


@pytest.fixture
def simple_parts_random_stage_1_complete(
    simple_parts_lpg, simple_parts_random_stage_1_instances
) -> dict:
    scg = simple_parts_lpg.get_projection("Part Definition")

    random_generator_playbook_phase_1_singletons(
        simple_parts_lpg.model,
        scg,
        simple_parts_random_stage_1_instances,
    )

    return simple_parts_random_stage_1_instances


@pytest.fixture
def simple_parts_random_stage_2_complete(
    simple_parts_lpg, simple_parts_random_stage_1_complete
) -> dict:
    scg = simple_parts_lpg.get_projection("Part Definition")

    random_generator_playbook_phase_2_rollup(scg, simple_parts_random_stage_1_complete)

    random_generator_playbook_phase_2_unconnected(
        simple_parts_lpg.model, simple_parts_random_stage_1_complete
    )

    return simple_parts_random_stage_1_complete


@pytest.fixture
def simple_parts_random_stage_3_complete(
    simple_parts_lpg, simple_parts_random_stage_2_complete
) -> dict:
    feature_sequences = build_sequence_templates(lpg=simple_parts_lpg)

    random_generator_playbook_phase_3(
        simple_parts_lpg.model,
        feature_sequences,
        simple_parts_random_stage_2_complete,
    )

    return simple_parts_random_stage_2_complete


@pytest.fixture
def simple_actions_client() -> SysML2Client:
    return simple_actions_model_loaded_client()


@pytest.fixture
def simple_actions_lpg() -> SysML2LabeledPropertyGraph:
    new_lpg = SysML2LabeledPropertyGraph()
    client = simple_actions_model_loaded_client()

    new_lpg.model = client.model

    return new_lpg


@pytest.fixture
def kerbal_model() -> pm.Model:
    return pm.Model.load_from_file(FIXTURES / "Kerbal.json")


@pytest.fixture
def all_models() -> Dict[Path, pm.Model]:
    return {
        path.name.replace(".json", ""): pm.Model.load_from_file(path.resolve())
        for path in FIXTURES.glob("*.json")
    }


@pytest.fixture
def simple_actions_stable_names():
    client = simple_actions_model_loaded_client()
    lpg = SysML2LabeledPropertyGraph()
    lpg.model = client.model
    return build_stable_id_lookups(lpg)


@pytest.fixture
def simple_actions_random_stage_1_instances(simple_actions_lpg) -> dict:
    ptg = simple_actions_lpg.get_projection("Part Typing")
    scg = simple_actions_lpg.get_projection("Part Definition")

    full_multiplicities = random_generator_phase_1_multiplicities(simple_actions_lpg, ptg, scg)

    return {
        type_id: create_set_with_new_instances(
            sequence_template=[simple_actions_lpg.model.elements[type_id]],
            quantities=[number],
        )
        for type_id, number in full_multiplicities.items()
    }


@pytest.fixture
def simple_actions_random_stage_1_complete(
    simple_actions_lpg, simple_actions_random_stage_1_instances
) -> dict:
    scg = simple_actions_lpg.get_projection("Part Definition")

    random_generator_playbook_phase_1_singletons(
        simple_actions_lpg.model,
        scg,
        simple_actions_random_stage_1_instances,
    )

    return simple_actions_random_stage_1_instances


@pytest.fixture
def simple_actions_random_stage_2_complete(
    simple_actions_lpg, simple_actions_random_stage_1_instances
) -> dict:
    scg = simple_actions_lpg.get_projection("Part Definition")

    random_generator_playbook_phase_2_rollup(scg, simple_actions_random_stage_1_instances)

    random_generator_playbook_phase_2_unconnected(
        simple_actions_lpg.model, simple_actions_random_stage_1_instances
    )

    return simple_actions_random_stage_1_instances


@pytest.fixture
def simple_actions_random_stage_3_complete(
    simple_actions_lpg, simple_actions_random_stage_2_complete
) -> dict:
    feature_sequences = build_sequence_templates(lpg=simple_actions_lpg)

    random_generator_playbook_phase_3(
        simple_actions_lpg.model,
        feature_sequences,
        simple_actions_random_stage_2_complete,
    )

    return simple_actions_random_stage_2_complete
