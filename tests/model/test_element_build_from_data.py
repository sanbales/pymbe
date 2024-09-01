from uuid import uuid4
import pymbe.api as pm

from pymbe.model import Element, Model

from pymbe.model_modification import new_element_ownership_pattern

def test_generate_element_from_dictionary():
    empty_model = pm.Model(elements={})

    package_model_namespace_data = {
        'aliasIds': [],
        'isImpliedIncluded': False,
        '@type': "Namespace",
        '@id': str(uuid4()),
        'ownedRelationship': []
    }
    package_model_data = {
        'name': "Example Builder Model",
        'isLibraryElement': False,
        'filterCondition': [],
        'ownedElement': [],
        'owner': {},
        '@type': "Package",
        '@id': str(uuid4()),
        'ownedRelationship': []
    }
    new_ns = Element.new(data=package_model_namespace_data,model=empty_model)
    new_package = Element.new(data=package_model_data,model=empty_model)
    new_element_ownership_pattern(
        owner=new_ns, ele=new_package, model=empty_model, member_kind="OwningMembership"
    )

    partdefinition_data = empty_model.metamodel.pre_made_dicts["PartDefinition"]
    partdefinition_data.update({"@type": "PartDefinition", "declaredName": "Demo Unit", "@id": str(uuid4())})

    partdefinition_ele = Element.new(data=partdefinition_data, model=empty_model)

    assert partdefinition_ele.declaredName == "Demo Unit"

def test_add_relationship():

    empty_model = pm.Model(elements={})

    package_model_namespace_data = {
        'aliasIds': [],
        'isImpliedIncluded': False,
        '@type': "Namespace",
        '@id': str(uuid4()),
        'ownedRelationship': []
    }
    package_model_data = {
        'name': "Example Builder Model",
        'isLibraryElement': False,
        'filterCondition': [],
        'ownedElement': [],
        'owner': {},
        '@type': "Package",
        '@id': str(uuid4()),
        'ownedRelationship': []
    }
    new_ns = Element.new(data=package_model_namespace_data,model=empty_model)
    new_package = Element.new(data=package_model_data,model=empty_model)
    new_element_ownership_pattern(
        owner=new_ns, ele=new_package, model=empty_model, member_kind="OwningMembership"
    )

    partdefinition_data = empty_model.metamodel.pre_made_dicts["PartDefinition"]
    partusage_data = empty_model.metamodel.pre_made_dicts["PartUsage"]
    fm_data = empty_model.metamodel.pre_made_dicts["FeatureMembership"]

    partdefinition_data.update({"@type": "PartDefinition", "declaredName": "Demo Unit", "@id": str(uuid4())})
    partusage_data.update({"@type": "PartUsage", "declaredName": "Demo Component", "@id": str(uuid4())})
    fm_data.update({"source": [{"@id": partdefinition_data["@id"]}],
                "target": [{"@id": partusage_data["@id"]}],
                "owningRelatedElement": {"@id": partdefinition_data["@id"]},
               "@id": str(uuid4()),
                "@type": "FeatureMembership"})

    partdefinition_ele = Element.new(data=partdefinition_data, model=empty_model)
    partusage_ele = Element.new(data=partusage_data, model=empty_model)
    fm_ele = Element.new(data=fm_data, model=empty_model)

    empty_model._add_relationship(fm_ele)

    assert partdefinition_ele.throughFeatureMembership[0] == partusage_ele
    assert partusage_ele.reverseFeatureMembership[0] == partdefinition_ele