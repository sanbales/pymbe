import traceback
import typing as ty
from functools import lru_cache
from pathlib import Path
from warnings import warn

import networkx as nx
import traitlets as trt
from ruamel.yaml import YAML

from ..model import Element, Model

yaml = YAML(typ="unsafe", pure=True)


class SysML2LabeledPropertyGraph(trt.HasTraits):  # pylint: disable=too-many-instance-attributes
    """A Labelled Property Graph for SysML v2.

    ..todo::
        Integrate it with the RDF representation.
    """

    FILTERED_DATA_KEYS: tuple = ("@context",)
    ATTRIBUTE_TO_EDGES: tuple = (
        # Adds additional relationships (edges) based on element attributes
        # dict(attribute="owner", metatype="Owned", reversed=True),
    )

    sysml_projections: dict = trt.Dict()

    model: Model = trt.Instance(Model, allow_none=True)
    graph: nx.MultiDiGraph = trt.Instance(nx.MultiDiGraph, args=tuple())

    max_graph_size: int = trt.Int(default_value=256)
    merge: bool = trt.Bool(default_value=False)

    nodes: dict = trt.Dict()
    edges: dict = trt.Dict()

    node_types: tuple = trt.Tuple()
    edge_types: tuple = trt.Tuple()

    nodes_by_type: dict = trt.Dict()
    edges_by_type: dict = trt.Dict()

    def __repr__(self):
        return (
            "<SysML v2 LPG: "
            f"{len(self.graph.nodes):,d} nodes, "
            f"{len(self.graph.edges):,d} edges"
            ">"
        )

    def __getitem__(self, *args):
        return self.graph.__getitem__(*args)

    @trt.observe("graph", "max_graph_size")
    def _update_projections(self, *_):
        projections = yaml.load(
            stream=(Path(__file__).parent / "sysml_subgraphs.yml").read_text(),
        )
        # TODO: Look into filtering the other projections
        if len(self.graph) > self.max_graph_size:
            projections.pop("Complete", None)

        self.sysml_projections = dict(projections)

    @trt.observe("model")
    def update(self, change: trt.Bunch):
        self._adapt.cache_clear()

        model = change.new
        if not isinstance(model, Model):
            return

        all_relationships: ty.Set[Element] = set(model.all_relationships.values())

        # Draw as edges all non-abstract relationships
        graph_relationships: ty.Set[Element] = {
            relationship
            for relationship in all_relationships
            if "isAbstract" not in relationship._data
        }
        # And everything else is a node
        graph_elements: ty.Set[Element] = set(model.elements.values()).difference(
            graph_relationships
        )

        # Add the abstract_relationships edges we removed from relationships
        expanded_relationships: ty.Set[Element] = all_relationships.difference(graph_relationships)
        edges_from_abstract_relationships = [
            [
                relationship._id,
                related_element_entry["@id"],
                f"""{relationship._metatype}End""",
                {
                    "@id": f"{relationship._id}_{endpt_index}",
                    "@type": f"""{relationship._metatype}End""",
                    "RelationshipType": True,
                },
            ]
            for relationship in expanded_relationships
            for endpt_index, related_element_entry in enumerate(
                relationship._data["relatedElement"]
            )
        ]

        graph = nx.MultiDiGraph()
        if self.merge:
            graph.add_nodes_from(self.graph)
            graph.add_edges_from(self.graph)

        old_graph = self.graph
        del old_graph

        graph.add_nodes_from({element._id: element._data for element in graph_elements}.items())

        graph.add_edges_from(
            [
                [
                    source._id,  # source node (str id)
                    target._id,  # target node (str id)
                    relationship._metatype,  # edge metatype (str name)
                    relationship._data,  # edge data (dict)
                ]
                for relationship in graph_relationships
                for source in relationship.source
                for target in relationship.target
            ]
            + edges_from_abstract_relationships
        )

        with self.hold_trait_notifications():
            self.nodes = dict(graph.nodes)
            self.edges = dict(graph.edges)

            self.node_types = tuple(
                sorted({node["@type"] for node in self.nodes.values() if "@type" in node})
            )
            self.edge_types = tuple(
                sorted({edge["@type"] for edge in self.edges.values() if "@type" in edge})
            )

            self.nodes_by_type = {
                node_type: [
                    node for node in graph.nodes if graph.nodes[node].get("@type") == node_type
                ]
                for node_type in self.node_types
            }
            self.edges_by_type = {
                edge_type: [edge for edge in graph.edges if edge[2] == edge_type]
                for edge_type in self.edge_types
            }

            self.graph = graph

    def get_projection_instructions(self, projection: str) -> dict:
        instructions = {**self.sysml_projections.get(projection, {})}
        if not instructions:
            raise ValueError(
                f"Could not find SysML Project: '{projection}'.\n"
                f"Options available are: {tuple(self.sysml_projections)}"
            )

        for key in tuple(instructions):
            # revert included for excluded types
            if key.startswith("included_") and key.endswith("_types"):
                included_types = instructions.pop(key)
                types_key = key.replace("included_", "")
                types = set(getattr(self, types_key)).difference(included_types)
                instructions[f"excluded_{types_key}"] = tuple(sorted(types))

        function_attributes = self.adapt.__annotations__
        return {key: value for key, value in instructions.items() if key in function_attributes}

    def get_implied_edges(self, *implied_edge_types):
        from .edge_generators import IMPLIED_GENERATORS  # pylint: disable=import-outside-toplevel

        new_edges = []
        for implied_edge_type in implied_edge_types:
            edge_generator = IMPLIED_GENERATORS.get(implied_edge_type)
            if edge_generator is None:
                warn(f"Could not find an implied edge generator for '{implied_edge_type}'")
                continue
            new_edges += edge_generator(lpg=self)
        return new_edges

    def get_projection(
        self,
        projection: str,
        packages: ty.Optional[ty.Union[ty.List[Element], ty.Tuple[Element]]] = None,
    ) -> nx.Graph:
        if isinstance(packages, Element):
            packages = [packages]
        return self.adapt(
            **self.get_projection_instructions(
                projection=projection,
            ),
            included_packages=packages or [],
        )

    def adapt(
        self,
        excluded_node_types: ty.Union[list, set, tuple] = None,
        excluded_edge_types: ty.Union[list, set, tuple] = None,
        reversed_edge_types: ty.Union[list, set, tuple] = None,
        implied_edge_types: ty.Union[list, set, tuple] = None,
        included_packages: ty.Union[list, set, tuple] = None,
    ) -> ty.Union[nx.Graph, nx.DiGraph]:
        """
        Using the existing graph, filter by node and edge types, and/or
        reverse certain edge types.
        """
        excluded_edge_types = excluded_edge_types or []
        excluded_node_types = excluded_node_types or []
        reversed_edge_types = reversed_edge_types or []
        implied_edge_types = implied_edge_types or []
        included_packages = included_packages or []

        # NOTE: Sorting into a tuple to make the LRU Cache work
        return self._adapt(
            excluded_edge_types=tuple(sorted(excluded_edge_types)),
            excluded_node_types=tuple(sorted(excluded_node_types)),
            reversed_edge_types=tuple(sorted(reversed_edge_types)),
            implied_edge_types=tuple(sorted(implied_edge_types)),
            included_packages=tuple(sorted(included_packages)),
        ).copy()

    @lru_cache(maxsize=1024)
    def _adapt(
        self,
        excluded_node_types: ty.Union[list, set, tuple] = None,
        excluded_edge_types: ty.Union[list, set, tuple] = None,
        reversed_edge_types: ty.Union[list, set, tuple] = None,
        implied_edge_types: ty.Union[list, set, tuple] = None,
        included_packages: ty.Union[list, set, tuple] = None,
    ) -> nx.Graph:
        graph = self.graph.copy()
        new_edges = self.get_implied_edges(*implied_edge_types)
        if new_edges:
            graph.add_edges_from(new_edges)

        mismatched_node_types = set(excluded_node_types).difference(self.node_types)
        if mismatched_node_types:
            print(f"These node types are not in the graph: {mismatched_node_types}.")

        mismatched_edge_types = {
            *excluded_edge_types,
            *reversed_edge_types,
        }.difference(self.edge_types)
        if mismatched_edge_types:
            print(f"These edge types are not in the graph: {mismatched_edge_types}.")

        included_nodes = sum(
            [
                nodes
                for node_type, nodes in self.nodes_by_type.items()
                if node_type not in excluded_node_types
            ],
            [],
        )
        if included_packages:
            all_elements = self.model.elements
            included_nodes = [
                node_id
                for node_id in included_nodes
                if all_elements[node_id] in included_packages
                or any(all_elements[node_id].is_in_package(pkg) for pkg in included_packages)
            ]
        subgraph = graph.__class__(graph.subgraph(included_nodes))

        def _process_edge(src, tgt, typ, data, rev_types):
            data = {**self.graph.edges.get((src, tgt, typ), {}), **data}
            if typ in rev_types:
                tgt, src = src, tgt
                typ += "^-1"
                data["@type"] = typ
            return src, tgt, typ, data

        edges = [
            _process_edge(src, tgt, typ, data, reversed_edge_types)
            for (src, tgt, typ), data in subgraph.edges.items()
            if typ not in excluded_edge_types
        ]

        new_graph = graph.__class__()
        new_graph.add_edges_from(edges)
        new_graph.update(
            nodes={node: self.graph.nodes.get(node) for node in new_graph.nodes}.items()
        )
        return new_graph

    @staticmethod
    def _make_undirected(graph):
        if graph.is_multigraph():
            return nx.MultiGraph(graph)
        return nx.Graph(graph)

    def get_path_graph(
        self,
        graph: nx.Graph,
        source: str,
        target: str,
        enforce_directionality: bool = True,
        try_reverse: bool = True,
    ):  # pylint: disable=too-many-arguments
        """Make a new graph with the shortest paths between two nodes"""
        new_graph = graph if enforce_directionality else self._make_undirected(graph)
        try:
            nodes = set(sum(map(list, nx.all_shortest_paths(new_graph, source, target)), []))
            return graph.__class__(graph.subgraph(nodes))

        except (nx.NetworkXError, nx.NetworkXException):
            warn(traceback.format_exc())
            if try_reverse:
                return self.get_path_graph(
                    graph=graph,
                    source=target,
                    target=source,
                    enforce_directionality=enforce_directionality,
                    try_reverse=False,
                )
            return graph.__class__()

    def get_spanning_graph(
        self,
        graph: nx.Graph,
        seeds: ty.Union[list, set, tuple],
        max_distance: int = 2,
        enforce_directionality: bool = True,
    ):
        if not enforce_directionality:
            graph = self._make_undirected(graph)
        seed_nodes = {id_: max_distance for id_ in seeds if id_ in graph.nodes}
        seed_elements = [
            self.model.elements[id_]
            for id_ in seeds
            if id_ not in seed_nodes and id_ in self.model.elements
        ]
        seed_edges = [
            (element.source, element.target)
            for element in seed_elements
            if element._is_relationship
        ]

        distances = {node: max_distance - 1 for node in set(sum(seed_edges, [])) if node in graph}
        distances.update(seed_nodes)

        max_iter = 100
        while sum(distances.values()) > 0 and max_iter > 0:
            max_iter -= 1
            for node_id, distance in tuple(distances.items()):
                if distance < 1:
                    continue
                for neighbor in graph.neighbors(node_id):
                    distances[neighbor] = distances.get(neighbor, distance - 1)
                distances[node_id] -= 1

        nodes = tuple(distances)

        return graph.__class__(graph.subgraph(nodes))
