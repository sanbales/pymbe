# pylint: disable=redefined-outer-name
from pathlib import Path
from typing import Dict

import pytest

import pymbe
import pymbe.api as pm
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
def kerbal_model() -> pm.Model:
    return get_model("Kerbal")


@pytest.fixture
def kerbal_ids_by_type(kerbal_model) -> dict:
    return {
        metatype: [element._id for element in elements]
        for metatype, elements in kerbal_model.ownedMetatype.items()
    }


@pytest.fixture
def kerbal_stable_names(kerbal_model):
    return build_stable_id_lookups(SysML2LabeledPropertyGraph(model=kerbal_model))


@pytest.fixture()
def all_kerbal_names(kerbal_model) -> list:
    return [
        element._data["name"]
        for element in kerbal_model.elements.values()
        if "name" in element._data
    ]


@pytest.fixture
def kerbal_lpg(kerbal_model) -> SysML2LabeledPropertyGraph:
    return SysML2LabeledPropertyGraph(model=kerbal_model)


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
def simple_parts_model() -> pm.Model:
    return get_model("Simple Parts Model")


@pytest.fixture
def simple_parts_lpg(simple_parts_model) -> SysML2LabeledPropertyGraph:
    return SysML2LabeledPropertyGraph(model=simple_parts_model)


@pytest.fixture
def simple_parts_stable_names(simple_parts_model):
    return build_stable_id_lookups(SysML2LabeledPropertyGraph(model=simple_parts_model))


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
def simple_actions_model() -> pm.Model:
    return get_model("Simple Action Example")


@pytest.fixture
def simple_actions_lpg(simple_actions_model) -> SysML2LabeledPropertyGraph:
    return SysML2LabeledPropertyGraph(model=simple_actions_model)


@pytest.fixture
def all_models() -> Dict[Path, pm.Model]:
    return {
        path.name.replace(".json", ""): pm.Model.load_from_file(path.resolve())
        for path in FIXTURES.glob("*.json")
    }


@pytest.fixture
def simple_actions_stable_names(simple_actions_model):
    return build_stable_id_lookups(SysML2LabeledPropertyGraph(simple_actions_model))


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
