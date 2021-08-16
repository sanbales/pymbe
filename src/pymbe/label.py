# TODO: Refactor this whole thing and integrate it better with the new Model approach
# Module for computing useful labels and signatures for SysML v2 elements
from .model import Element, Model

DEFAULT_MULTIPLICITY_LIMITS = dict(lower="0", upper="*")


def get_label(element: Element) -> str:
    metatype = element._metatype
    model = element._model
    data = element._data
    name = data.get("name")

    # Get the element type(s)
    types: list = data.get("type") or []
    if isinstance(types, dict):
        types = [types]
    type_names = [model.elements[type_["@id"]]._data.get("name") for type_ in types]
    type_names = [str(type_name) for type_name in type_names if type_name]
    value = element._data.get("value")
    if name:
        if type_names:
            # TODO: look into using other types (if there are any)
            name += f": {type_names[0]}"
        return name
    if value and metatype.startswith("Literal"):
        metatype = type_names[0] if type_names else metatype.replace("Literal", "Occurred Literal")
        return f"{value} «{metatype}»"
    if metatype == "MultiplicityRange":
        return get_label_for_multiplicity(multiplicity=element)
    if metatype.endswith("Expression"):
        return get_label_for_expression(
            expression=element,
            type_names=type_names,
        )
    if "@id" in data:
        return f"""{data["@id"]} «{metatype}»"""

    return "blank"


def get_label_for_id(element_id: str, model: Model) -> str:
    return get_label(model.elements[element_id])


def get_label_for_expression(
    expression: Element,
    type_names: list,
) -> str:
    metatype = expression._metatype
    if metatype not in (
        "Expression",
        "FeatureReferenceExpression",
        "InvocationExpression",
        "OperatorExpression",
        "PathStepExpression",
        "NullExpression",
    ):
        raise NotImplementedError(f"Cannot create M1 signature for: {metatype}")

    if metatype == "NullExpression":
        return "{}"

    if metatype == "FeatureReferenceExpression":
        try:
            referent = expression.referent
            referent_name = referent.name
            name_chain = referent.qualifiedName.split("::")
            index = 0
            while not referent_name and index < len(name_chain):
                index += 1
                referent_name = name_chain[-index]
                if referent_name.lower() == "null":
                    referent_name = None
        except (AttributeError, KeyError, TypeError):
            referent_name = "UNNAMED"
        return f"FRE.{referent_name}"

    prefix = ""
    if "input" in expression._data:
        inputs = [
            expression._model.elements[an_input["@id"]] for an_input in expression._data["input"]
        ]
    else:
        inputs = []
    if isinstance(inputs, Element):
        inputs = [inputs]
    input_names = [an_input.name for an_input in inputs if an_input.name]
    try:
        result: Element = expression.result
    except AttributeError:
        result = None

    if result and metatype == "Expression":
        parameter_members = [result] + inputs
        # Scan memberships to find non-parameter members
        non_parameter_members = [
            get_label(element=owned_member)
            for owned_member in expression.ownedMember
            if owned_member not in parameter_members
        ]
        prefix = non_parameter_members[0] if non_parameter_members else ""
    elif metatype == "OperatorExpression":
        prefix = expression["operator"]
    elif metatype == "InvocationExpression":
        prefix = type_names[0] if type_names else ""
    elif metatype == "PathStepExpression":
        path_step_names = []
        for owned_fm in expression.ownedFeatureMembership:
            if owned_fm._metatype == "FeatureMembership":
                member = owned_fm.memberElement
                # expect FRE here
                refered = member.get("referent")
                if refered:
                    path_step_names.append(refered.get("name") or refered._id)

        prefix = ".".join(path_step_names)
    inputs = f""" ({", ".join(input_names)})""" if input_names else ""
    if result is None:
        return f"""{prefix}{inputs} => {"Unnamed Result"}"""
    return f"""{prefix}{inputs} => {result.name or "Unnamed Result"}"""


def get_label_for_multiplicity(multiplicity: Element) -> str:
    values = {}
    data = multiplicity._data
    for limit, default in DEFAULT_MULTIPLICITY_LIMITS.items():
        literal_id = (data.get(f"{limit}Bound") or {}).get("@id")
        if literal_id is None:
            values[limit] = default
            continue
        value = multiplicity._model.elements.get(literal_id, {})._data.get("value")
        if value is None:
            value = default
        values[limit] = value
    return f"""{values["lower"]}..{values["upper"]}"""


def get_qualified_label(element: Element, parameter_name_map: dict, start: bool) -> str:
    earlier_name = "Model"

    element_data = element._data
    all_elements = element._model.elements

    if "owner" in element_data:
        if element_data["owner"] is not None:
            element_owner = all_elements[element_data["owner"]["@id"]]
            earlier_name = get_qualified_label(element_owner, parameter_name_map, False)
    else:
        name = element_data["name"]
        return f"{name}"

    printed_name = ""
    if element_data["@id"] in parameter_name_map:
        printed_name = parameter_name_map[element_data["@id"]]
    else:
        printed_name = get_label(element)

    tail = ""
    if start:
        tail = " <<" + element_data["@type"] + ">>"

    earlier_name = f"{earlier_name}::{printed_name}{tail}"

    return earlier_name
