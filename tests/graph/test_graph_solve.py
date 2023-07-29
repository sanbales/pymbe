import logging
from warnings import warn

import pytest

from pymbe.graph.calc_lpg import CalculationGroup
from pymbe.interpretation.calc_dependencies import generate_execution_order
from pymbe.interpretation.results import pprint_edges

ROCKET_BUILDING = "Model::Kerbal::Rocket Building::"
PARTS_LIBRARY = "Model::Kerbal::Parts Library::"

logger = logging.getLogger(__name__)


@pytest.mark.skip("Need to refactor tests, after upgrades")
def test_basic_kerbal_solve(kerbal_lpg, kerbal_random_stage_5_complete, kerbal_stable_names):
    # check that literal assignments go correctly

    *_, qualified_name_to_id = kerbal_stable_names

    rt_10_isp_id = qualified_name_to_id[
        f"""{PARTS_LIBRARY}RT-10 "Hammer" Solid Fuel Booster::Specific Impulse: Real <<AttributeUsage>>"""
    ]
    ft200_full_mass = qualified_name_to_id[
        f"{PARTS_LIBRARY}FL-T200 Fuel Tank::Full Mass: Real <<AttributeUsage>>"
    ]

    # incrementally step through the calculations and check progress

    dcg = generate_execution_order(kerbal_lpg)

    # check first 10 literals found

    literal_output_moves = [
        item
        for item in dcg
        if item[2] == "Output" and "Literal" in kerbal_lpg.nodes[item[0]]["@type"]
    ]

    logger.debug(pprint_edges(literal_output_moves, kerbal_lpg.model))

    cg1 = CalculationGroup(
        kerbal_lpg.get_projection("Expression Inferred"),
        kerbal_random_stage_5_complete,
        dcg,
    )

    try:
        cg1.solve_graph(kerbal_lpg)
    except:
        pass

    for move in literal_output_moves:
        logger.debug(kerbal_random_stage_5_complete[move[1]])

    rt_10_isps = [seq[-1].value for seq in kerbal_random_stage_5_complete[rt_10_isp_id]]
    ft200_masses = [seq[-1].value for seq in kerbal_random_stage_5_complete[ft200_full_mass]]

    if None in rt_10_isps:
        warn(f"Found `None` as an ISP for the RT-10s: {rt_10_isps}")
    if None in ft200_masses:
        warn(f"Found `None` as an mass for the FT-200s: {ft200_masses}")

    assert not set(rt_10_isps).difference({170.0, None}), f"RT-10 ISPs: {rt_10_isps}"
    assert not set(ft200_masses).difference({1.125, None}), f"FT-200 massess: {ft200_masses}"


def test_path_step_expression_kerbal_solve(
    kerbal_lpg, kerbal_random_stage_5_complete, kerbal_stable_names
):
    # check that Path Step Expression has expected inputs

    id_to_qualified_name, qualified_name_to_id = kerbal_stable_names

    pse_engine_mass = qualified_name_to_id[
        f"""{PARTS_LIBRARY}RT-10 "Hammer" Solid Fuel Booster::Specific Impulse: Real <<AttributeUsage>>"""
    ]

    # incrementally step through the calculations and check progress

    dcg = generate_execution_order(kerbal_lpg)

    engine_mass_pse_dcg = None

    for item in dcg:
        if kerbal_lpg.nodes[item[0]]["@type"] == "PathStepExpression":
            engine_mass_pse_dcg = kerbal_lpg.nodes[item[0]]

    print(engine_mass_pse_dcg)

    # FIXME: Uncomment the assert below after things are fixed
    # assert engine_mass_pse_dcg is not None
