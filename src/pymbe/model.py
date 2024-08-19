import json
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Collection, Dict, List, Optional, Tuple, Union
from uuid import uuid4
from warnings import warn

<<<<<<< HEAD
from pymbe.metamodel import MetaModel
=======
from pymbe.metamodel import MetaModel, derive_attribute, list_relationship_metaclasses
from pymbe.query.metamodel_navigator import get_effective_basic_name
>>>>>>> bjorn/main

OWNER_KEYS = ("owner", "owningRelatedElement", "owningRelationship")
VALUE_METATYPES = ("AttributeDefinition", "AttributeUsage", "DataType")


def is_id_item(item):
    return isinstance(item, dict) and item["@id"] is not None and isinstance(item["@id"], str)


class ListOfNamedItems(list):
    """A list that also can return items by their name."""

    # FIXME: figure out why __dir__ of returned objects think they are lists
    def __getitem__(self, key):
        item_map = {
            item._data["declaredName"]: item
            for item in self
            if isinstance(item, Element) and "declaredName" in item._data
        }
        effective_item_map = {
            item._data["effectiveName"]: item
            for item in self
            if isinstance(item, Element) and "effectiveName" in item._data
        }
        if key in item_map:
            return item_map[key]
        if key in effective_item_map:
            return effective_item_map[key]
        if isinstance(key, int):
            return super().__getitem__(key)
        return None


class Naming(Enum):
    """An enumeration for how to repr SysML elements"""

    IDENTIFIER = "identifier"
    LONG = "long"
    QUALIFIED = "qualified"
    SHORT = "short"
    LABEL = "label"
    BASIC = "basic"

    def get_name(self, element: "Element") -> str:
        naming = self._value_  # pylint: disable=no-member

        # TODO: Check with Bjorn, he wanted: (a)-[r:RELTYPE {name: a.name + '<->' + b.name}]->(b)
        if element._is_relationship:
            name_and_meta = " ".join(
                (get_effective_basic_name(element), f"«{element._metatype}»")
            ).strip()
            return f"{name_and_meta} ({element.source} ←→ {element.target})"

        data = element._data
        if naming == Naming.QUALIFIED.value:
            return f"""<{data["qualifiedName"]}>"""

        if naming == Naming.IDENTIFIER.value:
            return f"""<{data["@id"]}>"""
        if naming == Naming.LABEL.value:
            return element._derived.get("label")
        name = data.get("declaredName") or data.get("value") or data["@id"]
        if naming == Naming.SHORT.value:
            return f"<{name}>"

        # TODO: Bring basic naming over here (and include member name from owning memberships)

        return f"""<{name} «{data["@type"]}»>"""


class ModelClient:
    def get_element_data(self, element_id: str) -> dict:
        raise NotImplementedError("Must be implemented by the subclass")


@dataclass(repr=False)
class Model:  # pylint: disable=too-many-instance-attributes
    """A SysML v2 Model"""

    # TODO: Look into making elements immutable (e.g., frozen dict)
    elements: Dict[str, "Element"]

    name: str = "SysML v2 Model"

    all_relationships: Dict[str, "Element"] = field(default_factory=dict)
    all_non_relationships: Dict[str, "Element"] = field(default_factory=dict)

    ownedElement: ListOfNamedItems = field(  # pylint: disable=invalid-name
        default_factory=ListOfNamedItems,
    )
    ownedMetatype: Dict[str, List["Element"]] = field(  # pylint: disable=invalid-name
        default_factory=dict,
    )
    ownedRelationship: List["Element"] = field(  # pylint: disable=invalid-name
        default_factory=list,
    )

    max_multiplicity = 10

    source: Any = None

    _api: ModelClient = None
    _initializing: bool = True
    _naming: Naming = Naming.LABEL  # The scheme to use for retrieving element names
    _labeling: Naming = Naming.LABEL  # The scheme to use for repr'ing the elements

    metamodel: Optional[MetaModel] = None

    _referenced_models: List["Model"] = field(  # pylint: disable=invalid-name
        default_factory=list,
    )

    _metamodel_hints: Dict[str, List[List[str]]] = field(
        default_factory=dict
    )  # hints about attribute primary v derived, expected value type, etc.

    def __post_init__(self):
        self.metamodel = self.metamodel or MetaModel()
        self._metamodel_hints = self.metamodel.metamodel_hints

        # self.elements = {
        #     id_: Element(
        #         _data={**data, "@id": id_},
        #         _model=self,
        #         _metamodel_hints={
        #             att_key: att_data
        #             for att_key, att_data in self._metamodel_hints[data["@type"]].items()
        #         },
        #     )
        #     for id_, data in self.elements.items()
        #     if isinstance(data, dict)
        # }

        self.elements = {
            id_: Element(
                _data={**data, "@id": id_},
                _model=self,
                _metamodel_hints=self._metamodel_hints[data["@type"]],
            )
            for id_, data in self.elements.items()
            if isinstance(data, dict)
        }

        self._add_owned()

        # Modify and add derived data to the elements
        self._add_relationships()
        self._add_labels()
        self._initializing = False

    def __repr__(self) -> str:
        data = self.source or f"{len(self.elements)} elements"
        return f"<SysML v2 Model ({data})>"

    @staticmethod
    def load(
        elements: Collection[Dict],
        **kwargs,
    ) -> "Model":
        """Make a Model from an iterable container of elements"""
        return Model(
<<<<<<< HEAD
            elements={element["@id"]: element for element in elements if "@id" in element},
=======
            elements={
                element.get("identity", element).get("@id"): element.get("payload", element)
                for element in elements
            },
>>>>>>> bjorn/main
            **kwargs,
        )

    @staticmethod
    def load_from_file(filepath: Union[Path, str], encoding: str = "utf-8") -> "Model":
        """Make a model from a JSON file"""
        if isinstance(filepath, str):
            filepath = Path(filepath)

        if not filepath.is_file():
            raise ValueError(f"'{filepath}' does not exist!")

        return Model.load(
            elements=json.loads(filepath.read_text(encoding=encoding)),
            name=filepath.name,
            source=filepath.resolve(),
        )

    @staticmethod
    def load_from_post_file(filepath: Union[Path, str], encoding: str = "utf-8") -> "Model":
        """Make a model from a JSON file formatted to POST to v2 API (includes payload fields)"""
        if isinstance(filepath, str):
            filepath = Path(filepath)

        if not filepath.is_file():
            raise ValueError(f"'{filepath}' does not exist!")

        with open(filepath, "r", encoding="UTF-8") as raw_post_fp:
            element_raw_post_data = json.load(raw_post_fp)

            factored_data = []
            for raw_post in element_raw_post_data:
                factored_data_element = dict(raw_post["payload"].items()) | {
                    "@id": raw_post["identity"]["@id"]
                }
                factored_data.append(factored_data_element)

        return Model.load(
            elements=factored_data,
            name=filepath.name,
            source=filepath.resolve(),
        )
    
    @staticmethod
    def load_from_mult_post_files(filepath_list: List, encoding: str = "utf-8") -> "Model":
        """Make a model from multiple JSON files formatted to POST to v2 API (includes payload fields)"""
        factored_data = []

        for filepath in filepath_list:
            if isinstance(filepath, str):
                filepath = Path(filepath)

            if not filepath.is_file():
                raise ValueError(f"'{filepath}' does not exist!")

            with open(filepath, "r", encoding="UTF-8") as raw_post_fp:
                element_raw_post_data = json.load(raw_post_fp)

                for raw_post in element_raw_post_data:
                    factored_data_element = dict(raw_post["payload"].items()) | {
                        "@id": raw_post["identity"]["@id"]
                    }
                    factored_data.append(factored_data_element)

        return Model.load(
            elements=factored_data,
            name=filepath.name,
            source=filepath_list[0].resolve(),
        )

    @property
    def packages(self) -> Tuple["Element", ...]:
        return tuple(
            element for element in self.elements.values() if element._metatype == "Package"
        )

    def get_element(
        self, element_id: str, fail: bool = True, resolve: bool = True
    ) -> Optional["Element"]:
        """Get an element, or retrieve it from the API if it is there"""
        element = self.elements.get(element_id)
        if element and not isinstance(element, Element):
            return element
        if not element and self._api:
            data = self._api.get_element_data(element_id) if resolve else {}
            element = Element(
                _id=element_id,
                _data=data,
                _model=self,
                _metamodel_hints=self._metamodel_hints[data["@type"]],
            )
        if element and resolve and element._is_proxy:
            element.resolve()
        if fail and element is None:
            # hacky way to use library references, need more scaleable version
            for ref_model in self._referenced_models:
                try:
                    element = ref_model.get_element(element_id)
                    return element
                except KeyError:
                    pass
            raise KeyError(f"Could not retrieve '{element_id}' from the API")
        return element

    def _add_element(self, element: "Element") -> "Element":
        id_ = element._id
        metatype = element._metatype

        self.elements[id_] = element

        if element.get_owner() is None:
            if element not in self.ownedElement:
                self.ownedElement += [element]

        if metatype not in self.ownedMetatype:
            self.ownedMetatype[metatype] = []
        if element not in self.ownedMetatype[metatype]:
            self.ownedMetatype[metatype] += [element]

        if element._is_relationship:
            if element not in self.ownedRelationship:
                self.ownedRelationship += [element]
            if id_ not in self.all_relationships:
                self.all_relationships[id_] = element
        else:
            if id_ not in self.all_non_relationships:
                self.all_non_relationships[id_] = element

        if not self._initializing:
            self._add_labels(element)
        return element

    def save_to_file(
        self, filepath: Union[Path, str] = None, indent: int = 2, encoding: str = "utf-8"
    ):
        filepath = filepath or self.name
        if not self.elements:
            warn("Model has no elements, nothing to save!")
            return
        if isinstance(filepath, str):
            filepath = Path(filepath)
        if not filepath.name.endswith(".json"):
            filepath = filepath.parent / f"{filepath.name}.json"
        if filepath.exists():
            warn(f"Overwriting {filepath}")
        filepath.write_text(
            json.dumps(
                [element._data for element in self.elements.values() if element._data],
                indent=indent,
            ),
            encoding=encoding,
        )

    def _add_labels(self, *elements):
        """Attempts to add a label to the elements"""
        from .label import get_label  # pylint: disable=import-outside-toplevel

        elements = elements or self.elements.values()
        for element in elements:
            label = get_label(element=element)
            if label:
                element._label = label

    def _add_owned(self):
        """Adds owned elements, relationships, and metatypes to the model"""
        elements = self.elements

        self.all_relationships = {
            id_: element for id_, element in elements.items() if element._is_relationship
        }
        self.all_non_relationships = {
            id_: element for id_, element in elements.items() if not element._is_relationship
        }

        owned = [element for element in elements.values() if element.get_owner() is None]
        self.ownedElement = ListOfNamedItems(
            element for element in owned if not element._is_relationship
        )

        self.ownedRelationship = [
            relationship for relationship in owned if relationship._is_relationship
        ]

        by_metatype = defaultdict(list)
        for element in elements.values():
            by_metatype[element._metatype].append(element)
        self.ownedMetatype = dict(by_metatype)

    def _add_relationships(self):
        """Adds relationships to elements"""

        # TODO: make this more elegant...  maybe.
        for relationship in self.all_relationships.values():
            self._add_relationship(relationship)

    def _add_relationship(self, relationship):
        relationship_mapper = {
            "through": ("source", "target"),
            "reverse": ("target", "source"),
        }

        try:
            endpoints = {
                endpoint_type: [
                    self.get_element(endpoint["@id"]) for endpoint in relationship._data[endpoint_type]
                ]
                for endpoint_type in ("source", "target")
            }
        except KeyError as likley_id_error:
            warn(str(likley_id_error) + f" call was from a relation of type {relationship._metatype}.")
            return
            
        except TypeError:
            # may be malformed with just one source and target - this compensates for that case
            endpoints = {}
            source_value = relationship._data["source"]
            target_value = relationship._data["target"]

            if isinstance(source_value, dict):
                endpoints.update({"source": [self.get_element(relationship._data["source"]["@id"])]
                                  })
            else:
                endpoints.update({"source": [
                    self.get_element(endpoint["@id"]) for endpoint in relationship._data["source"]
                ]}
                )

            if isinstance(target_value, dict):
                endpoints.update({"target": [self.get_element(relationship._data["target"]["@id"])]
                                  })
            else:
                endpoints.update({"target": [
                    self.get_element(endpoint["@id"]) for endpoint in relationship._data["target"]
                ]}
                )
            return
            
        metatype = relationship._metatype
        for direction, (key1, key2) in relationship_mapper.items():
            endpts1, endpts2 = endpoints[key1], endpoints[key2]
            for endpt1 in endpts1:
                for endpt2 in endpts2:
                    endpt1._derived[f"{direction}{metatype}"] += [{"@id": endpt2._data["@id"]}]

    def reference_other_model(self, ref_model: "Model"):
        if ref_model not in self._referenced_models:
            self._referenced_models.append(ref_model)


@dataclass(repr=False)
class Element:  # pylint: disable=too-many-instance-attributes
    """A SysML v2 Element"""

    _data: Dict[str, Any]
    _model: Model

    _metamodel_hints: Dict[str, List[str]]

    _id: str = field(default_factory=lambda: str(uuid4()))
    _label: str = None
    _metatype: str = "Element"
    _derived: Dict[str, List] = field(default_factory=lambda: defaultdict(list))
    # TODO: replace this with instances sequences
    # _instances: List["Instance"] = field(default_factory=list)
    _is_abstract: bool = False
    _is_proxy: bool = True
    _is_relationship: bool = False
    _package: "Element" = None

    # TODO: Add comparison to allow sorting of elements (e.g., by name and then by id)

    def __post_init__(self):
        # if not self._model._initializing:
        #    self._model.elements[self._id] = self
        if self._data:
            self.resolve()

    def resolve(self):
        if not self._is_proxy:
            return

        model = self._model

        if not self._data:
            if not model._api:
                raise SystemError("Model must have an API to retrieve the data from!")
            self._data = model._api.get_element_data(self._id)
        data = self._data
        self._id = data["@id"]
        self._metatype = data["@type"]

        self._is_abstract = bool(data.get("isAbstract"))
        self._is_relationship = bool(data.get("source")) and bool(data.get("target"))
        for key, items in data.items():
            # set up owned elements to be referencable by their name
            if key.startswith("owned") and isinstance(items, list):
                data[key] = ListOfNamedItems(items)
            # add Pythonic property to Element object based on metamodel for primary data values
            elif (
                key in self._metamodel_hints
                and self._metamodel_hints[key][1] == "primary"
                and self._metamodel_hints[key][3] != "EReference"
            ):
                setattr(self, key, items)
        if not model._initializing:
            self._model._add_element(self)
        self._is_proxy = False

    def __call__(self, *args, **kwargs):
        element = kwargs.pop("element", None)
        if element:
            warn("When instantiating an element, you cannot pass it element.")

    def __dir__(self):
        return sorted(
            set(
                list(super().__dir__())
                + [key for key in [*self._data, *self._derived] if key.isidentifier()]
            )
        )

    def __eq__(self, other):
        if isinstance(other, str):
            return self._id == other
        # TODO: Assess if this is more trouble than it's worth
        # if isinstance(other, dict):
        #     return tuple(sorted(self._data.items())) == tuple(sorted(other.items()))
        return self is other

    # TODO: Make this safe for through and reverse by return empty collection is no key found
    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(f"Cannot find {key}") from exc

    def __getitem__(self, key: str):
        found = False
        for source in ("_data", "_derived"):
            source = self.__getattribute__(source)
            if key in source:
                if key in self._metamodel_hints and self._metamodel_hints[key][1] == "primary":
                    found = True
                    item = source[key]
                    break

                if key in self._metamodel_hints and self._metamodel_hints[key][1] == "derived":
                    break
                found = True
                item = source[key]
                break
            if key[7:] in list_relationship_metaclasses():
                found = True
                item = []
        if not found:
            if (
                key in self._metamodel_hints
                and self._metamodel_hints[key][1] == "derived"
                and self._metamodel_hints[key][3] == "EReference"
            ):
                found = True
                item = self.derive_attribute(key)

        if not found:
            raise KeyError(f"No '{key}' in {self}")

        if isinstance(item, (dict, str)):
            item = self.__safe_dereference(item)
        elif isinstance(item, (list, tuple, set)):
            items = [self.__safe_dereference(sub_item) for sub_item in item]
            return type(item)(items)
        return item

    def __hash__(self):
        id_ = self._id
        if id_ is None:
            warn(
                f"Element (oid={id(self)}, data={self._data}) has no '@id'! Generating one for it"
            )
            id_ = self._data["@id"] = str(uuid4())
        return hash(id_)

    def __lt__(self, other):
        if isinstance(other, str):
            other = self._model.get_element(other, fail=False) or other
        if not isinstance(other, Element):
            raise ValueError(f"Cannot compare an element to {type(other)}")
        if self.get("name", None) and other.get("name", None):
            return self.name < other.name
        return self._id < other._id

    def __repr__(self):
        self_str = self._model._labeling.get_name(element=self)
        if self_str is None:
            return "No Name"

        return self_str

    @property
    def basic_name(self) -> str:
        # TODO: Move this over to the naming class and get result back
        return (
            self._data.get("declaredName")
            or self._data.get("name")
            or self._data.get("effectiveName")
            or ""
        )

    @property
    def label(self) -> str:
        return self._label

    @property
    def relationships(self) -> Dict[str, Any]:
        return {
            key: self[key]
            for key in self._derived
            if key.startswith("through") or key.startswith("reverse")
        }

    @property
    def owning_package(self) -> "Element":
        """A lazy property to remember what package elements belongs to."""
        if self._package is None:
            owner = self.get_owner()
            while owner and owner._metatype != "Package":
                owner = owner.get_owner()
            self._package = owner
        return self._package

    @lru_cache(maxsize=1024)
    def is_in_package(self, package: "Element") -> bool:
        owning_package = self.owning_package
        while owning_package:
            if owning_package == package:
                return True
            owning_package = owning_package.owning_package
        return False

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def get_element(self, element_id) -> "Element":
        return self._model.get_element(element_id)

    def get_owner(self) -> "Element":
        data = self._data
        owner_id = None
        for key in OWNER_KEYS:
            owner_id = (data.get(key) or {}).get("@id")
            if owner_id is not None:
                break
        if owner_id is None:
            return None
        try:
            return self._model.get_element(owner_id)
        except KeyError as no_owner:
            if str(self) != "No Name":
                raise KeyError(
                    f"Failed to find element with id {owner_id} while "
                    + f"looking for owner of {self}"
                ) from no_owner

            playback_name = str(self)
            if "declaredName" in data.keys():
                playback_name = data["declaredName"]
            raise KeyError(
                f"Failed to find element with id {owner_id} while "
                + f"looking for owner of {playback_name}"
            ) from no_owner

    @staticmethod
    def new(data: dict, model: Model) -> "Element":
        return Element(
            _data=data, _model=model, _metamodel_hints=model._metamodel_hints[data["@type"]]
        )

    def __safe_dereference(self, item):
        """If given a reference to another element, try to get that element"""
        try:
            if isinstance(item, dict) and "@id" in item:
                if len(item) > 1:
                    warn(f"Found a reference with more than one entry: {item}")
                item = item["@id"]
            return self._model.get_element(item)
        except KeyError:
            return item

    def derive_attribute(self, key: str) -> Union["Element", Tuple["Element"]]:
        # entry point for deriving attributes within elements on demand
        attribute_derivations_by_key = dict(
            type=self.derive_type,
            ownedMember=self.derive_owned_member,
        )
        deriver = attribute_derivations_by_key.get(key)
        if deriver:
            return deriver()
        if "owned" in key:
            return self.derive_owned_x(key[5:])
        raise NotImplementedError(f"The method to derive '{key}' has yet to be developed.")

    def derive_type(self) -> Tuple["Element"]:
        if hasattr(self, "throughFeatureTyping"):
            return self.throughFeatureTyping
        return ()

    def derive_owned_member(self) -> Tuple["Element"]:
        # TODO: Make this work with generalization of metatypes
        return tuple(
            owned_related_ele
            for owned_rel in self.ownedRelationship
            for owned_related_ele in owned_rel.ownedRelatedElement
            if owned_rel._metatype == "OwningMembership"
        )

    def derive_owned_x(self, owned_kind: str) -> Tuple["Element"]:
        return tuple(
            owned_related_ele
            for owned_rel in self.ownedRelationship
            for owned_related_ele in owned_rel.ownedRelatedElement
            if owned_related_ele._metatype == owned_kind
        )
