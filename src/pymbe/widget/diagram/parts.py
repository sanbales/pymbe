from ipyelk.elements import (
    Compartment,
    ElementMetadata,
    Node,
    NodeProperties,
    Port,
    PortProperties,
    Record,
)
from pydantic import Field


class PartMetadata(ElementMetadata):
    sysml_id: str = Field(default="NO_ID")


class NodeUsage(Node):
    """An instance of a part usage."""

    properties: NodeProperties = Field(
        default_factory=lambda *_: NodeProperties(cssClasses="usage")
    )


class PortUsage(Port):
    """An instance of a port usage."""

    properties: PortProperties = Field(
        default_factory=lambda *_: PortProperties(cssClasses="usage")
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
        metatype = data.get("@type")
        label = data.get("value") or data.get("label") or data.get("name") or id_
        kwargs = dict(metadata=PartMetadata(sysml_id=id_))

        if isinstance(label, str):
            label = label.replace(f"«{metatype}»", "").strip()

        if metatype in ("MultiplicityRange",) or metatype.startswith("Literal"):
            width = int(width / 2)

        part = Part(data=data, id=id_, width=width, **kwargs)
        part.title = Compartment(**kwargs).make_labels(
            headings=[
                f"«{metatype}»",
                f"{label}",
            ],
        )

        # TODO: add properties
        properties = []
        if properties:
            part.attrs = Compartment(**kwargs).make_labels(
                headings=["properties"],
                content=[cls.make_property_label(prop) for prop in properties],
            )
        return part
