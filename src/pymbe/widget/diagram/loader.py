import typing as ty

import networkx as nx
import traitlets as trt
from ipyelk import ElementLoader, MarkElementWidget
from ipyelk.elements import layout_options as opt

from .part_diagram import PartDiagram
from .parts import Part
from .relationships import Relationship


class SysmlLoader(ElementLoader):  # pylint: disable=abstract-method
    """A customized loader for SysML LPGs."""

    all_parts: ty.Dict[str, Part] = trt.Dict(
        key_trait=trt.Unicode(),
        value_trait=trt.Instance(Part),
        help="A store of all the parts, i.e., nodes, that have been displayed.",
    )
    all_relationships: ty.Dict[str, Relationship] = trt.Dict(
        key_trait=trt.Unicode(),
        value_trait=trt.Instance(Relationship),
        help="A store of all the relationships, i.e., edges, that have been displayed.",
    )
    part_diagram: PartDiagram = trt.Instance(PartDiagram, args=())

    # pylint: disable=no-self-use
    @trt.default("default_node_opts")
    def _make_node_opts(self):
        return opt.OptionsWidget(
            options=[
                opt.Direction(value="RIGHT"),
                opt.HierarchyHandling(),
            ]
        ).value

    # pylint: disable=no-self-use
    @trt.default("default_label_opts")
    def _make_label_opts(self):
        return opt.OptionsWidget(
            options=[
                opt.NodeLabelPlacement(horizontal="center", vertical="center"),
            ]
        ).value

    def clear(self, clear_part_diagram=True):
        self.all_parts = {}
        self.all_relationships = {}

        # Clear the part diagram
        if clear_part_diagram:
            part_diagram = self.part_diagram
            part_diagram.children = []
            part_diagram.edges = []
            part_diagram.labels = []
            part_diagram.ports = []

    def load_from_graphs(self, new: nx.Graph, old: nx.Graph = None) -> MarkElementWidget:
        new = nx.Graph() if new in (None, trt.Undefined) else new
        old = nx.Graph() if old in (None, trt.Undefined) else old

        part_diagram = self.part_diagram

        # TODO: figure a way to not have the nuke from orbit
        self.clear()

        # Get SysML IDs for nodes and edges that need to be managed
        # old_nodes, new_nodes = set(old.nodes), set(new.nodes)
        # old_edges = {data["@id"] for data in old.edges.values()}
        # new_edges = {data["@id"] for data in new.edges.values()}

        # exiting_nodes = old_nodes.difference(new_nodes)
        # exiting_relationships = old_edges.difference(new_edges)
        # entering_relationships = new_edges.difference(old_edges)

        all_parts = self.all_parts
        new_parts = {
            sysml_id: Part.from_data(node_data)
            for sysml_id, node_data in new.nodes.items()
            if node_data and sysml_id not in all_parts
        }
        all_parts.update(new_parts)

        # TODO: Look into managing entering/exiting relationships
        # for parent, relationship in part_diagram.all_relationships:
        #     # if relationship.metadata.sysml_id in exiting_relationships:
        #     parent.edges.remove(relationship)

        all_relationships = self.all_relationships
        new_relationships = {
            data["@id"]: part_diagram.add_relationship(
                source=all_parts[source],
                target=all_parts[target],
                cls=Relationship.from_metatype(metatype),
                data=data,
            )
            for (source, target, metatype), data in new.edges.items()
            if source in all_parts and target in all_parts
            # and data["@id"] in entering_relationships
        }
        all_relationships.update(new_relationships)

        # for parent, part in part_diagram.all_parts:
        #     if part.metadata.sysml_id in exiting_nodes:
        #         parent.children.remove(part)

        return self.load(root=part_diagram)
