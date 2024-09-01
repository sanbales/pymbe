from uuid import NAMESPACE_URL, uuid5

from pymbe.model import Element, Model
from pymbe.model_modification import build_from_classifier_pattern


def id_for_quantity(name: str):
    url = "https://www.dumbdata.org/" + name
    return uuid5(NAMESPACE_URL, url)


def build_indefinite_boundaries(
    library_model: Model, target_package: Element, active_model: Model
):

    # find Occurrence in libraries

    occurrence_ns = [
        library_model_ns
        for library_model_ns in library_model.ownedElement
        if library_model_ns.throughOwningMembership[0].declaredName == "Occurrences"
    ][0]

    ocurrences_eles = occurrence_ns.throughOwningMembership[0].throughOwningMembership

    occurrence = None

    for ocurrences_ele in ocurrences_eles:
        if ocurrences_ele._metatype == "Class":
            if hasattr(ocurrences_ele, "declaredName"):
                if ocurrences_ele.declaredName == "Occurrence":
                    occurrence = ocurrences_ele

    indefinite_time = build_from_classifier_pattern(
        owner=target_package,
        name="Indefinite Time",
        model=active_model,
        metatype="Class",
        superclasses=[occurrence],
        specific_fields={"ownedRelationship": []},
    )

    indefinite_space = build_from_classifier_pattern(
        owner=target_package,
        name="Indefinite Space",
        model=active_model,
        metatype="Class",
        superclasses=[occurrence],
        specific_fields={"ownedRelationship": []},
    )

    indefinite_life = build_from_classifier_pattern(
        owner=target_package,
        name="Indefinite Life",
        model=active_model,
        metatype="Class",
        superclasses=[occurrence],
        specific_fields={"ownedRelationship": []},
    )

    # print(f"Created indefinite time {indefinite_time} and indefinite space {indefinite_space}")

    return (indefinite_time, indefinite_space, indefinite_life)
