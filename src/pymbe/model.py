import json
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union
from uuid import uuid4
from warnings import warn

OWNER_KEYS = ("owner", "owningRelatedElement", "owningRelationship")
VALUE_METATYPES = ("AttributeDefinition", "AttributeUsage", "DataType")


class ListOfNamedItems(list):
    """A list that also can return items by their name."""

    # FIXME: figure out why __dir__ of returned objects think they are lists
    def __getitem__(self, key):
        item_map = {
            item._data["name"]: item
            for item in self
            if isinstance(item, Element) and "name" in item._data
        }
        if key in item_map:
            return item_map[key]
        return super().__getitem__(key)


class Naming(Enum):
    """An enumeration for how to repr SysML elements"""

    IDENTIFIER = "identifier"
    LONG = "long"
    QUALIFIED = "qualified"
    SHORT = "short"

    def get_name(self, element: "Element") -> str:
        naming = self._value_  # pylint: disable=no-member

        # TODO: Check with Bjorn, he wanted: (a)-[r:RELTYPE {name: a.name + '<->' + b.name}]->(b)
        if element._is_relationship:
            return f"<{element._metatype}({element.source} ←→ {element.target})>"

        data = element._data
        if naming == Naming.QUALIFIED:
            return f"""<{data["qualifiedName"]}>"""

        if naming == Naming.IDENTIFIER:
            return f"""<{data["@id"]}>"""

        name = data.get("name") or data["@id"]
        if naming == Naming.SHORT:
            return f"<{name}>"
        return f"""<{name} «{data["@type"]}»>"""


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

    max_multiplicity = 100

    source: Any = None

    _naming: Naming = Naming.LONG  # The scheme to use for repr'ing the elements

    def __post_init__(self):
        self.elements = {
            id_: Element(_data=data, _model=self)
            for id_, data in self.elements.items()
            if isinstance(data, dict)
        }

        self._add_owned()

        # Modify and add derived data to the elements
        self._add_relationships()
        self._add_labels()

    def __repr__(self) -> str:
        data = self.source or f"{len(self.elements)} elements"
        return f"<SysML v2 Model ({data})>"

    @staticmethod
    def load(
        elements: Union[List[Dict], Set[Dict], Tuple[Dict]],
        **kwargs,
    ) -> "Model":
        """Make a Model from an iterable container of elements"""
        return Model(
            elements={element["@id"]: element for element in elements},
            **kwargs,
        )

    @staticmethod
    def load_from_file(filepath: Union[Path, str]) -> "Model":
        """Make a model from a JSON file"""
        if isinstance(filepath, str):
            filepath = Path(filepath)

        if not filepath.is_file():
            raise ValueError(f"'{filepath}' does not exist!")

        return Model.load(
            elements=json.loads(filepath.read_text()),
            name=filepath.name,
            source=filepath.resolve(),
        )

    @property
    def packages(self) -> Tuple["Element"]:
        return tuple(
            element for element in self.elements.values() if element._metatype == "Package"
        )

    def save_to_file(self, filepath: Union[Path, str], indent: int = 2) -> bool:
        if isinstance(filepath, str):
            filepath = Path(filepath)
        filepath.write_text(
            json.dumps(
                [element._data for element in self.elements.values()],
                indent=indent,
            ),
        )

    def _add_labels(self):
        """Attempts to add a label to the elements"""
        from .label import get_label  # pylint: disable=import-outside-toplevel

        for element in self.elements.values():
            label = get_label(element=element)
            if label:
                element._derived["label"] = label

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
        relationship_mapper = {
            "through": ("source", "target"),
            "reverse": ("target", "source"),
        }
        # TODO: make this more elegant...  maybe.
        for relationship in self.all_relationships.values():
            endpoints = {
                endpoint_type: [
                    self.elements[endpoint["@id"]]
                    for endpoint in relationship._data[endpoint_type]
                ]
                for endpoint_type in ("source", "target")
            }
            metatype = relationship._metatype
            for direction, (key1, key2) in relationship_mapper.items():
                endpts1, endpts2 = endpoints[key1], endpoints[key2]
                for endpt1 in endpts1:
                    for endpt2 in endpts2:
                        endpt1._derived[f"{direction}{metatype}"] += [{"@id": endpt2._data["@id"]}]


@dataclass(repr=False)
class Element:  # pylint: disable=too-many-instance-attributes
    """A SysML v2 Element"""

    _data: dict
    _model: Model

    _id: str = field(default_factory=lambda: str(uuid4()))
    _metatype: str = "Element"
    _derived: Dict[str, List] = field(default_factory=lambda: defaultdict(list))
    # TODO: replace this with instances sequences
    # _instances: List["Instance"] = field(default_factory=list)
    _is_abstract: bool = False
    _is_relationship: bool = False
    _package: "Element" = None

    # TODO: Add comparison to allow sorting of elements (e.g., by name and then by id)

    def __post_init__(self):
        if not self._data:
            self._is_proxy = True
        else:
            self.resolve()

    def resolve(self):
        if not self._data:
            raise NotImplementedError("Need to add functionality to get data for the element")
        data = self._data
        self._id = data["@id"]
        self._metatype = data["@type"]

        self._is_abstract = bool(data.get("isAbstract"))
        self._is_relationship = "relatedElement" in data
        for key, items in data.items():
            if key.startswith("owned") and isinstance(items, list):
                data[key] = ListOfNamedItems(items)
        self._is_proxy = False

    def __call__(self, *args, **kwargs):
        element = kwargs.pop("element", None)
        if element:
            warn("When instantiating an element, you cannot pass it element.")
        if self._metatype in VALUE_METATYPES:
            return ValueHolder(*args, **kwargs, element=self)
        return Instance(*args, **kwargs, element=self)

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
                found = True
                item = source[key]
                break
        if not found:
            raise KeyError(f"No '{key}' in {self}")

        if isinstance(item, (dict, str)):
            item = self.__safe_dereference(item)
        elif isinstance(item, (list, tuple, set)):
            items = [self.__safe_dereference(subitem) for subitem in item]
            return type(item)(items)
        return item

    def __hash__(self):
        return hash(self._data["@id"])

    def __lt__(self, other):
        if isinstance(other, str):
            if other in self._model.elements:
                other = self._model.elements[other]
        if not isinstance(other, Element):
            raise ValueError(f"Cannot compare an element to {type(other)}")
        if self.get("name", None) and other.get("name", None):
            return self.name < other.name
        return self._id < other._id

    def __repr__(self):
        return self._model._naming.get_name(element=self)

    @property
    def is_proxy(self):
        return not self._data

    @property
    def label(self) -> str:
        return self._derived.get("label")

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
        return self._model.elements.get(element_id)

    def get_owner(self) -> "Element":
        data = self._data
        owner_id = None
        for key in OWNER_KEYS:
            owner_id = (data.get(key) or {}).get("@id")
            if owner_id is not None:
                break
        if owner_id is None:
            return None
        return self._model.elements[owner_id]

    @staticmethod
    def new(data: dict, model: Model) -> "Element":
        return Element(_data=data, _model=model)

    def __safe_dereference(self, item):
        """If given a reference to another element, try to get that element"""
        try:
            if isinstance(item, dict) and "@id" in item:
                if len(item) > 1:
                    warn(f"Found a reference with more than one entry: {item}")
                item = item["@id"]
            return self._model.elements[item]
        except KeyError:
            return item


@dataclass
class Instance:
    """
    An M0 instantiation in the M0 universe (i.e., real things) interpreted from an M1 element.

    Sequences of instances are intended to follow the mathematical base semantics of SysML v2.
    """

    element: Element
    number: int = None
    name: str = ""

    __instances = defaultdict(list)

    def __post_init__(self):
        siblings = self.__instances[self.element]
        if self.number is None:
            self.number = len(siblings) + 1
        siblings += [self]
        if not self.name:
            element = self.element
            name = element.label or element._id
            self.name = f"{name}#{self.number}"

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(id(self))


@dataclass
class ValueHolder(Instance):
    """An M0 instantiation of a Value element"""

    value: Any = None

    def __repr__(self):
        value = self.value
        if value is None:
            value = "unset"
        return f"{self.name} ({value})"


InstanceDictType = Dict[str, List[List[Instance]]]
