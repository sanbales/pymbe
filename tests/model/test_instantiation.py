import pytest

from pymbe.model import VALUE_METATYPES, ValueHolder


@pytest.mark.skip("Need to refactor tests, after upgrades")
def test_instantiation(kerbal_model):
    model = kerbal_model

    for element in model.elements.values():
        label = element.label
        if not label or element._metatype not in VALUE_METATYPES:
            continue

        instance = element(value=1.5)
        assert isinstance(instance, ValueHolder)
        assert instance.value == 1.5
        repred = str(instance)
        assert element.name in repred
        assert "1.5" in repred

        unset_value_holder = element()
        assert unset_value_holder.value is None
        repred = str(unset_value_holder)
        assert element.name in repred
        assert "unset" in repred
