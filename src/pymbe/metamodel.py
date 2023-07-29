import json
from dataclasses import field
from importlib import resources as lib_resources
from typing import Any, Dict, List

# TODO: Is there a way to restore type hints for Element without inducing a circular dependency?


class MetaModel:
    """
    A class to hold meta-model information and support model modification actions
    """

    metamodel_hints: Dict[str, List[List[str]]] = field(default_factory=dict)

    pre_made_dicts: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __init__(self):
        self.pre_made_dicts = {}
        self._load_metahints()
        for metaclass in self.metamodel_hints.keys():
            self._load_template_data(metaclass_name=metaclass)

    def _load_metahints(self):
        """Load data file to get attribute hints"""
        ecore_atts = {}
        ecore_refs = {}

        with lib_resources.open_text("pymbe.static_data", "sysml_ecore_atts.json") as sysml_ecore:
            ecore_atts = json.load(sysml_ecore)
        with lib_resources.open_text(
            "pymbe.static_data", "sysml_ecore_derived_refs.json"
        ) as sysml_ecore_refs:
            ecore_refs = json.load(sysml_ecore_refs)

        # keys should be the same since they are all identified metaelements from ecore
        self.metamodel_hints = {k: ecore_atts[k] + ecore_refs[k] for k in ecore_atts.keys()}

    def _load_template_data(self, metaclass_name: str):
        local_hints = self.metamodel_hints[metaclass_name]

        data_template = {}

        for hint in local_hints:
            starter_field = None
            if hint[2] == "primary":
                # TODO: Figure out why some boolean and string attributes have 0 to -1
                # rather than 1 to 1 multiplicity
                if (
                    int(hint[6]) > 1
                    or int(hint[6]) == -1
                    and not (hint[3] == "Boolean" or hint[3] == "String")
                ):
                    starter_field = []
                else:
                    # TODO: One other janky override
                    if hint[0] == "aliasIds":
                        starter_field = []
                    elif hint[3] == "Boolean":
                        starter_field = False
                    elif hint[3] == "String":
                        starter_field = ""
                    elif hint[3] == "Integer":
                        starter_field = 0

            data_template.update({hint[0]: starter_field})

        self.pre_made_dicts.update({metaclass_name: data_template})
