from collections import defaultdict
from pathlib import Path

import traitlets as trt

import pytest

import pymbe
from pymbe.client import SysML2Client
from pymbe.graph import SysML2LabeledPropertyGraph
from pymbe.interpretation.interp_playbooks import (
    build_expression_sequence_templates,
    build_sequence_templates,
    create_set_with_new_instances,
    random_generator_phase_1_multiplicities,
    random_generator_playbook_phase_1_singletons,
    random_generator_playbook_phase_2_unconnected,
    random_generator_playbook_phase_2_rollup,
    random_generator_playbook_phase_3,
    random_generator_playbook_phase_4,
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
            contents = (
                "\nContents:\n" +
                "\n".join(map(str, FIXTURES.glob("*")))
            )
        raise ValueError(
            f"Could not load: '{json_file.absolute()}'!\n"
            "Did you forget to run git submodules? If so, run:\n"
            "  git submodule update --init\n"
            f"pyMBE is here: {PYMBE_ROOT.absolute()}\n"
            f"{TESTS_ROOT} exists: {TESTS_ROOT.exists()}\n"
            f"{FIXTURES} exists: {fixtures_exists}"
            + contents
        )

    helper_client = SysML2Client()
    helper_client._load_from_file(json_file)

    return helper_client


def kerbal_model_loaded_client() -> SysML2Client:
    return get_client("Kerbal")


def simple_parts_model_loaded_client() -> SysML2Client:
    return get_client("Simple Parts Model")


def simple_actions_model_loaded_client() -> SysML2Client:
    helper_client = SysML2Client()

    base_path = Path(__file__).parent.absolute()
    print(base_path)
    file_name = str(base_path) + "\\data\\Simple Actions Model\\elements.json"

    helper_client._load_disk_elements(file_name)

    return helper_client


@pytest.fixture
def kerbal_client() -> SysML2Client:
    return kerbal_model_loaded_client()


@pytest.fixture
def kerbal_ids_by_type(kerbal_client) -> dict:
    ids_dict = defaultdict(list)
    all_elements = kerbal_client.elements_by_id

    for ele_id, ele in all_elements.items():
        ids_dict[ele["@type"]].append(ele_id)

    return ids_dict


@pytest.fixture
def kerbal_stable_names():
    client = kerbal_model_loaded_client()
    lpg = SysML2LabeledPropertyGraph()
    lpg.update(client.elements_by_id, False)
    return build_stable_id_lookups(client.elements_by_id, lpg)


@pytest.fixture
def kerbal_client() -> SysML2Client:
    return kerbal_model_loaded_client()


@pytest.fixture()
def all_kerbal_names(kerbal_client) -> list:
    names = []
    all_elements = kerbal_client.elements_by_id

    for ele in all_elements.values():
        if "name" in ele:
            names.append(ele["name"])

    return names


@pytest.fixture
def kerbal_lpg() -> SysML2LabeledPropertyGraph:
    new_lpg = SysML2LabeledPropertyGraph()
    client = kerbal_model_loaded_client()
    trt.link(
        (client, "elements_by_id"),
        (new_lpg, "elements_by_id"),
    )
    return new_lpg


@pytest.fixture
def kerbal_random_stage_1_instances(kerbal_client, kerbal_lpg) -> dict:
    ptg = kerbal_lpg.get_projection("Part Typing")
    scg = kerbal_lpg.get_projection("Part Definition")

    full_multiplicities = random_generator_phase_1_multiplicities(kerbal_client.elements_by_id, kerbal_lpg, ptg, scg)

    return {
        type_id: create_set_with_new_instances(
            sequence_template=[kerbal_lpg.nodes[type_id]],
            quantities=[number],
        )
        for type_id, number in full_multiplicities.items()
    }


@pytest.fixture
def kerbal_random_stage_1_complete(kerbal_client, kerbal_lpg, kerbal_random_stage_1_instances) -> dict:
    scg = kerbal_lpg.get_projection("Part Definition")

    random_generator_playbook_phase_1_singletons(
        kerbal_client.elements_by_id,
        kerbal_lpg,
        scg,
        kerbal_random_stage_1_instances,
    )

    return kerbal_random_stage_1_instances


@pytest.fixture
def kerbal_random_stage_2_complete(kerbal_client, kerbal_lpg, kerbal_random_stage_1_complete) -> dict:
    scg = kerbal_lpg.get_projection("Part Definition")

    random_generator_playbook_phase_2_rollup(kerbal_client.elements_by_id, scg, kerbal_random_stage_1_complete)

    random_generator_playbook_phase_2_unconnected(kerbal_lpg.nodes, kerbal_random_stage_1_complete)

    return kerbal_random_stage_1_complete


@pytest.fixture
def kerbal_random_stage_3_complete(kerbal_lpg, kerbal_random_stage_2_complete) -> dict:
    # TODO: Ask Bjorn if we need to bring this back, the phase 3 function only needed the LPG
    # ptg = kerbal_lpg.get_projection("Part Typing")
    all_elements = kerbal_lpg.nodes
    feature_sequences = build_sequence_templates(lpg=kerbal_lpg)

    random_generator_playbook_phase_3(
        feature_sequences,
        all_elements,
        kerbal_lpg,
        kerbal_random_stage_2_complete,
    )

    return kerbal_random_stage_2_complete


@pytest.fixture
def kerbal_random_stage_4_complete(kerbal_lpg, kerbal_random_stage_3_complete) -> dict:
    expr_sequences = build_expression_sequence_templates(lpg=kerbal_lpg)

    random_generator_playbook_phase_4(
        kerbal_lpg.nodes,
        expr_sequences,
        kerbal_lpg,
        kerbal_random_stage_3_complete,
    )

    return kerbal_random_stage_3_complete


@pytest.fixture
def simple_parts_client() -> SysML2Client:

    return simple_parts_model_loaded_client()


@pytest.fixture
def simple_parts_lpg() -> SysML2LabeledPropertyGraph:
    new_lpg = SysML2LabeledPropertyGraph()
    client = simple_parts_model_loaded_client()

    new_lpg.update(client.elements_by_id, False)

    return new_lpg


@pytest.fixture
def simple_actions_client() -> SysML2Client:

    return simple_actions_model_loaded_client()


@pytest.fixture
def simple_actions_lpg() -> SysML2LabeledPropertyGraph:
    new_lpg = SysML2LabeledPropertyGraph()
    client = simple_actions_model_loaded_client()

    new_lpg.update(client.elements_by_id, False)

    return new_lpg
