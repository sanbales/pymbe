from pathlib import Path
from uuid import uuid4

import pytest

from pymbe.model import Element, Model


def test_respect_of_sysml(all_models):
    for model in all_models.values():
        for element in model.elements.values():
            element_attrs = set(super(element.__class__, element).__dir__())
            derived_attrs = set(element._derived)
            sysml_attrs = set(element._data)
            common_attrs = tuple(sorted(element_attrs.intersection(sysml_attrs)))
            assert (
                not common_attrs
            ), f"Found Element attributes conflicting with SysML: {common_attrs}"
            common_attrs = tuple(sorted(derived_attrs.intersection(sysml_attrs)))
            assert (
                not common_attrs
            ), f"Found derived attributes conflicting with SysML: {common_attrs}"


def test_kerbal_model(kerbal_model):
    model = kerbal_model
    kerbal = model.ownedElement["Kerbal"]

    assert kerbal.name == "Kerbal", f"Element name should be 'Kerbal', not '{kerbal.name}'"
    assert kerbal == kerbal._id, "Element should equate to its id"

    my_rocket = kerbal(name="My Rocket")
    assert (
        my_rocket.name == "My Rocket"
    ), f"Instance should be called 'My Rocket', not '{my_rocket.name}'"

    assert (
        kerbal.ownedElement["Parts Library"]
        .ownedElement["FL-T200 Fuel Tank"]
        .ownedElement["Empty Mass"]
        .ownedElement[0]
        .value
        == 0.125
    )


def test_relationships(kerbal_model):
    model = kerbal_model

    rocket_part = None
    for element in model.elements.values():
        if element.name == "Kerbal Rocket Part":
            rocket_part = element
            break

    subclass = None
    for subclass in rocket_part.reverseSuperclassing:
        if subclass.name == "Parachute":
            break

    assert subclass, "Did not find the `Parachute`"
    assert rocket_part, "Did not find the `Kerbal Rocket Part`"
    assert subclass.throughSuperclassing[0].name == rocket_part.name


def test_edge_cases():
    name = f"model_{uuid4()}"
    filename = f"{name}.json"
    model = Model(elements={}, name=name)
    cwd = Path(".")

    with pytest.warns(UserWarning):
        model.save_to_file()
    assert not list(cwd.glob(filename)), f"Model should NOT have been saved! {list(cwd.glob('*'))}"

    empty_element = Element(_data={}, _model=model)
    assert empty_element._is_proxy, "Element should be marked as proxy!"

    model.save_to_file()
    saved_file = list(cwd.glob(filename))
    assert saved_file, "Model should have been saved with .json extension added!"

    with pytest.warns(UserWarning):
        model.save_to_file(filename)
    saved_file[0].unlink()


def test_package_references(kerbal_model):
    model = kerbal_model
    packages = model.packages

    assert packages, "Kerbal should have some packages!"

    a_package, sub_package = None, None
    for a_package in packages:
        sub_packages = [
            pkg
            for pkg in a_package.ownedElement
            if pkg._metatype == "Package" and pkg.ownedElement
        ]
        if sub_packages:
            sub_package = sub_packages[0]
            break

    assert a_package, "Kerbal model should have a package that has another package"
    assert sub_package, "Kerbal model should have a nested package"

    owned_element = sub_package.ownedElement[0]
    assert sub_package.ownedElement[owned_element.name] == owned_element

    assert (
        owned_element.owning_package == sub_package
    ), f"{owned_element} should know it is owned by {sub_package}"
    assert owned_element.is_in_package(a_package), f"{owned_element} should be in {a_package}"
    assert owned_element.is_in_package(sub_package), f"{owned_element} should be in {sub_package}"
    assert not a_package.is_in_package(
        owned_element
    ), f"{a_package} should NOT be owned by {owned_element}"
    assert not a_package.is_in_package(
        sub_package
    ), f"{a_package} should NOT be owned by {sub_package}"
    assert not sub_package.is_in_package(
        owned_element
    ), f"{sub_package} should NOT be owned by {owned_element}"

    assert (a_package > sub_package) == (a_package.name > sub_package.name)


def test_accessors(kerbal_model):
    model = kerbal_model
    for element in model.elements.values():
        if (
            element._metatype == "ReturnParameterMembership"
            and "LiteralInteger" in element.relatedElement[0]._metatype
        ):
            a_return_parameter_membership = element
            break
    assert isinstance(a_return_parameter_membership.relatedElement[0].value, int)

    assert (
        a_return_parameter_membership.target[0].reverseReturnParameterMembership[0]
        == a_return_parameter_membership.relatedElement[0]
    )

    value = a_return_parameter_membership.get("some_missing_key", "a default for something")
    assert value == "a default for something"

    assert a_return_parameter_membership.get("some_other_missing_key") is None

    source, target = a_return_parameter_membership.relatedElement
    assert source.throughReturnParameterMembership[0] == target
    assert target.reverseReturnParameterMembership[0] == source
