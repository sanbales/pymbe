import pytest

from pymbe.interpretation.calc_dependencies import generate_execution_order

ROCKET_BUILDING = "Model::Kerbal::Rocket Building::"
PARTS_LIBRARY = "Model::Kerbal::Parts Library::"


@pytest.mark.skip("Need to refactor tests, after upgrades")
def test_kerbal_calc_order(kerbal_lpg, kerbal_random_stage_5_complete, kerbal_stable_names):
    # check that the right number of values are created for their features

    *_, qualified_name_to_id = kerbal_stable_names

    liquid_stage_type = qualified_name_to_id[f"{ROCKET_BUILDING}Liquid Stage <<PartDefinition>>"]

    number_liquid_stages = len(kerbal_random_stage_5_complete[liquid_stage_type])

    if number_liquid_stages < 1:
        print(">>> Didn't make any liquid stages!!!")

    dcg = generate_execution_order(kerbal_lpg)

    # the execution order will be ordered for examination

    sum_1_result = qualified_name_to_id[
        f"{ROCKET_BUILDING}Liquid Stage::Full Mass: Real::+ (sum (engines.Mass"
        " (FRE.engines)), sum (tanks.Full Mass (FRE.tanks))) => $result::sum"
        " (tanks.Full Mass (FRE.tanks)) => $result::tanks.Full Mass (FRE.tanks) <<Feature>>"
    ]
    top_plus = qualified_name_to_id[
        f"{ROCKET_BUILDING}Liquid Stage::Full Mass: Real::+ (sum (engines.Mass "
        "(FRE.engines)), sum (tanks.Full Mass (FRE.tanks))) => "
        "$result <<OperatorExpression>>"
    ]

    sum_1_in_dcg = None
    plus_in_dcg = None

    for item in dcg:
        if item[0] == sum_1_result:
            sum_1_in_dcg = sum_1_result
        if item[0] == top_plus:
            plus_in_dcg = top_plus

    assert sum_1_in_dcg is not None
    assert plus_in_dcg is not None
    assert len(kerbal_random_stage_5_complete[sum_1_result]) == len(
        kerbal_random_stage_5_complete[liquid_stage_type]
    )
    assert len(kerbal_random_stage_5_complete[top_plus]) == len(
        kerbal_random_stage_5_complete[liquid_stage_type]
    )
