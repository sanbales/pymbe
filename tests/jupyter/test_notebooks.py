import pytest
from testbook import testbook


@pytest.mark.skip("Need to refactor tests, after upgrades")
@testbook("docs/tutorials/01-Basic.ipynb", execute=True)
def test_tutorial(tb):
    """Test the Tutorial notebook"""
    tree = tb.ref("tree")
    m1_diagram = tb.ref("m1_diagram")

    assert len(m1_diagram.model.elements) > 0

    assert len(m1_diagram.model.elements) == len(tree.model.elements)

    assert not set(m1_diagram.model.elements).symmetric_difference(set(tree.model.elements))


@pytest.mark.skip("Need to refactor tests, after upgrades")
@testbook("tests/jupyter/notebooks/Playbook Explorer Parts Test.ipynb", execute=True)
def test_simple_parts_explorer(tb):
    m0_interpretation = tb.ref("m0_interpretation")
    assert m0_interpretation


@testbook("docs/how_to_guides/SysML Model.ipynb", execute=True)
def test_sysml_model(tb):
    kerbal_element = tb.ref("kerbal")
    assert kerbal_element._data["name"] == "Kerbal"
