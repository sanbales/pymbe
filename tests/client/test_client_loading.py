from warnings import warn

import pytest
import requests

from pymbe.client import APIClient
from pymbe.model import Element

SYSML_SERVER_URL = "http://sysml2-sst.intercax.com"  # Alternative: sysml2.intercax.com


def can_connect(host: str, port: int = 9000):
    try:
        requests.get(f"{host}:{port}")
        return True
    except:  # pylint: disable=bare-except  # noqa: E722
        return False


def test_client_load_kerbal(kerbal_model):
    assert len(kerbal_model.elements) == 380


def test_client_load_find_names(all_kerbal_names):
    assert "Liquid Stage" in all_kerbal_names
    assert "$result" in all_kerbal_names
    assert "Empty Mass" in all_kerbal_names


def test_client_load_find_types(kerbal_ids_by_type):
    assert "PartDefinition" in kerbal_ids_by_type
    assert "FeatureTyping" in kerbal_ids_by_type
    assert "FeatureReferenceExpression" in kerbal_ids_by_type

    assert len(kerbal_ids_by_type["PartDefinition"]) == 17


@pytest.mark.skipif(
    not can_connect(host=SYSML_SERVER_URL),
    reason=f"Can't connect to {SYSML_SERVER_URL}",
)
def test_remote_connection():
    client = APIClient()
    client.host_url = SYSML_SERVER_URL

    assert client.projects

    client.page_size = 20

    client.selected_project = list(client.projects)[0]
    client.selected_commit = list(client._get_project_commits())[0]
    model = client.get_model()

    assert model.elements

    # Test resolving an element from the API
    element: Element = [
        el for el in model.ownedElement if el.get("name") and not el._is_relationship
    ][0]
    label = element.label
    element._data = {}
    element._derived = {}
    element._is_proxy = True
    model.ownedElement.remove(element)
    model.ownedMetatype.pop(element._metatype)
    model.all_non_relationships.pop(element._id)
    model.elements.pop(element._id)
    element.resolve()
    assert label == element.label
    assert element in model.ownedElement
    assert element in model.ownedMetatype[element._metatype]
    assert model.all_non_relationships[element._id] == element
    assert model.elements[element._id] == element

    owned_relationships: Element = [rel for rel in model.ownedElement if rel._is_relationship]
    if not owned_relationships:
        warn("Model does not have any owned relationships!")
    else:
        relationship = owned_relationships[0]
        label = relationship.label
        relationship._data = {}
        relationship._derived = {}
        relationship._is_proxy = True
        model.ownedElement.remove(relationship)
        model.ownedRelationship.remove(relationship)
        model.ownedMetatype[relationship._metatype].remove(relationship)
        model.all_relationships.pop(relationship._id)
        model.elements.pop(relationship._id)
        relationship.resolve()
        assert label == relationship.label
        assert relationship in model.ownedElement
        assert relationship in model.ownedMetatype[relationship._metatype]
        assert model.all_relationships[relationship._id] == relationship
        assert model.elements[relationship._id] == relationship
