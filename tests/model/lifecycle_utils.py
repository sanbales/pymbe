import logging
from dataclasses import dataclass, field
from typing import Any, Collection, Dict, List, Set, Tuple, Union

import pymbe.api as pm
from pymbe import Element, Model


class InstrumentedElement(Element):
    """
    A version of Element that logs its lifecycle to see how a modeling session interacts with it
    """

    def __post_init__(self):
        logging.info("[Element] entering post initialization.")

        # super().__post_init__()
        if not self._model._initializing:
            self._model.elements[self._id] = self
        if self._data:
            logging.info("[Element] data found. About to resolve.")
            self.resolve()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        logging.info(f"[Element] value of {name} is trying to be set to {value}")

    def resolve(self):
        if not self._is_proxy:
            logging.info("[Element] element is not proxy.")
            return

        logging.info("[Element] element is still in proxy mode - resolving internal data")

        model = self._model
        if not self._data:
            logging.info("[Element] no data found to resolve.")
            if not model._api:
                raise SystemError("Model must have an API to retrieve the data from!")
            self._data = model._api.get_element_data(self._id)
        data = self._data
        self._id = data["@id"]
        self._metatype = data["@type"]

        self._is_abstract = bool(data.get("isAbstract"))
        self._is_relationship = bool(data.get("relatedElement"))

        logging.info(
            f"[Element] assigned id = {self._id}, metatype = {self._metatype}, is abstract = {self._is_abstract}, and is relationship = {self._is_relationship}."
        )

        for key, items in data.items():
            if key.startswith("owned") and isinstance(items, list):
                logging.info(f"Key starting with owned found: {key}")
                data[key] = ListOfNamedItems(items)
                logging.info(f"Data resolved to: {data[key]}")
        if not model._initializing:
            self._model._add_element(self)
        self._is_proxy = False


@dataclass(repr=False)
class InstrumentedModel(Model):
    instrumented_name: str = "Bare Classed Feature"
    instrumented_element: InstrumentedElement = None

    def __post_init__(self):
        """Same as other code but with insertion of instrumented element"""

        self._load_metahints()

        logging.info(
            f"[Model] loading model with instrumented element with name '{self.instrumented_name}'"
        )

        instrumented_data = None
        instrumented_element = None
        for id_, data in self.elements.items():
            if "declaredName" in data and data["declaredName"] == self.instrumented_name:
                instrumented_data = (id_, data)

        logging.info(f"[Model] Model data is '{instrumented_data[1]}'")

        self.elements = {
            id_: Element(
                _data=data,
                _model=self,
                _metamodel_hints={att[0]: att[1:] for att in self._metamodel_hints[data["@type"]]},
            )
            for id_, data in self.elements.items()
            if isinstance(data, dict)
        }

        self.instrumented_element = InstrumentedElement(
            _data=instrumented_data[1],
            _model=self,
            _metamodel_hints={
                att[0]: att[1:] for att in self._metamodel_hints[instrumented_data[1]["@type"]]
            },
        )

        self.elements.update({instrumented_data[0]: self.instrumented_element})

        self._add_owned()

        # Modify and add derived data to the elements
        self._add_relationships()
        self._add_labels()
        self._initializing = False

    @staticmethod
    def load(
        elements: Collection[Dict],
        **kwargs,
    ) -> "Model":
        """Make a Model from an iterable container of elements"""
        return InstrumentedModel(
            elements={element["@id"]: element for element in elements},
            **kwargs,
        )

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
                    self.get_element(endpoint["@id"])
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
                        if endpt1 == self.instrumented_element:
                            logging.info(
                                f"[Model] updating derived field of instrument element for {direction}{metatype}"
                            )
