from pydantic import Field
from typing import Dict, Type, Union

from ipyelk.elements import (
    Compartment,
    Edge,
    Node,
    Partition,
    Port,
    Record,
    SymbolSpec,
    merge_excluded,
)

from .relationships import DirectedAssociation, Relationship
from .symbols import (
    make_arrow_symbol,
    make_containment_symbol,
    make_feature_typing_symbol,
    make_redefinition_symbol,
    make_rhombus_symbol,
    make_subsetting_symbol,
)


class Part(Record):
    """A container for storing the data of a SysML 2 Part."""

    data: dict = Field(
        default_factory=dict,
        description="The data in the part.",
    )
    id: str = Field(default="")

    @staticmethod
    def make_property_label(item):
        label = f"""- {item["name"]}"""
        if "type" in item:
            label += f""" :: {item["type"]}"""
        return label

    @classmethod
    def from_data(cls, data: dict, width=220):
        id_ = data["@id"]
        label = (
                data.get("value")
                or data.get("label")
                or data.get("name")
                or id_
        )
        metatype = data.get("@type")

        if (
                metatype in ("MultiplicityRange",)
                or metatype.startswith("Literal")
        ):
            width = int(width / 2)

        part = Part(data=data, id=id_, width=width)
        part.title = Compartment().make_labels(
            headings=[
                f"«{metatype}»",
                f"{label}",
            ],
        )

        # TODO: add properties
        properties = []
        if properties:
            part.attrs = Compartment().make_labels(
                headings=["properties"],
                content=[
                    cls.make_property_label(prop)
                    for prop in properties
                ],
            )
        return part


class PartDiagram(Partition):
    """A SysML 2 Part Diagram, based on the IPyELK BlockDiagram."""

    class Config:
        copy_on_model_validation = False
        excluded = merge_excluded(Partition, "symbols", "style")

    default_edge: Type[Edge] = Field(default=DirectedAssociation)

    symbols: SymbolSpec = SymbolSpec().add(
        make_arrow_symbol(identifier="generalization", r=5, closed=True),
        make_arrow_symbol(identifier="directed_association", r=5, closed=False),
        make_containment_symbol(identifier="containment", r=5),
        make_feature_typing_symbol(identifier="feature_typing", r=5),
        make_redefinition_symbol(identifier="redefinition", r=5),
        make_subsetting_symbol(identifier="subsetting", r=5),
        make_rhombus_symbol(identifier="composition", r=5),
        make_rhombus_symbol(identifier="aggregation", r=5),
    )

    style: Dict[str, Dict] = {
        # Elk Label styles for Box Titles
        " .elklabel.compartment_title_1": {
            # "font-weight": "bold",
        },
        " .elklabel.heading, .elklabel.compartment_title_2": {
            "font-weight": "bold",
        },
        # Style Arrowheads (future may try to )
        " .subsetting > .round > ellipse": {
            "fill": "var(--jp-elk-node-stroke)",
        },
        " .feature_typing > .round > ellipse": {
            "fill": "var(--jp-elk-node-stroke)",
        },
        " .internal > .elknode": {
            "stroke": "transparent",
            "fill": "transparent",
        },
        # Necessary for having the viewport use the whole vertical height
        " .lm-Widget.jp-ElkView .sprotty > .sprotty-root > svg.sprotty-graph": {
            "height": "unset!important",
        },
        " .dashed.elkedge > path ": {
            "stroke-dasharray": "3 3",
        },
        " text.elklabel.node_type_label": {
            "font-style": "italic",
        },
    }

    def add_relationship(
        self,
        source: Union[Node, Port],
        target: Union[Node, Port],
        cls: Type[Relationship] = Relationship,
        data: dict = None,
    ) -> Relationship:
        data = data or {}
        relationship = cls(source=source, target=target, data=data)
        self.edges.append(relationship)
        return relationship
