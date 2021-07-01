from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from uuid import uuid4
from warnings import warn

import networkx as nx
import ruamel.yaml as yaml
import traitlets as trt
import typing as ty

from ..core import Base


class SysML2LabeledPropertyGraph(Base):
    """A Labelled Property Graph for SysML v2.

    ..todo::
        Integrate it with the RDF representation.
    """

    FILTERED_DATA_KEYS: tuple = ("@context",)
    ATTRIBUTE_TO_EDGES: tuple = (
        # Adds additional relationships (edges) based on element attributes
        # dict(attribute="owner", metatype="Owned", reversed=True),
    )
    IMPLIED_GENERATORS: tuple = (
        # Adds additional implied edges based on a tuple of generators
        "get_implied_feedforward_edges",
    )
    sysml_projections: dict = trt.Dict()

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
            Loader=yaml.RoundTripLoader,
        )
        # TODO: Look into filtering the other projections
        if len(self.graph) > self.max_graph_size:
            projections.pop("Complete", None)

        self.sysml_projections = projections

    @staticmethod
    def _make_nx_multi_edge(source, target, metatype, **data):
        return (
            source,                           # source
            target,                           # target
            metatype,                         # edge type
            {                                 # edge data
                "@id": f"_{uuid4()}",
                "@type": metatype,
                "relatedElement": [
                    {"@id": source},
                    {"@id": target},
                ],
                "source": [{"@id": source}],
                "target": [{"@id": target}],
                **data
            },
        )

    def _make_new_edge(self, data: dict, edge_mapper: dict) -> list:
        attribute = edge_mapper["attribute"]
        if not data.get(attribute, {}):
            return []
        metatype = edge_mapper["metatype"]
        sources = [data["@id"]]
        targets = data[attribute]
        if isinstance(targets, (list, tuple, set)):
            targets = [
                t["@id"] for t in targets
            ]
            sources *= len(targets)
        elif isinstance(targets, dict):
            targets = [targets["@id"]]
        else:
            raise ValueError(
                f"Cannot process: {targets} of type: {type(targets)}."
            )
        if edge_mapper.get("reversed", False):
            sources, targets = targets, sources
        return [
            self._make_nx_multi_edge(source, target, metatype)
            for source, target in zip(sources, targets)
        ]

    def update(self, elements: dict, merge=None):
        self._adapt.cache_clear()

        merge = self.merge if merge is None else merge

        new_edges = [
            self._make_new_edge(data=element_data, edge_mapper=edge_mapper)
            for element_id, element_data in elements.items()
            for edge_mapper in self.ATTRIBUTE_TO_EDGES
        ]
        # flatten new edges
        new_edges = sum(new_edges, [])

        filtered_keys = list(self.FILTERED_DATA_KEYS) + [
            edge_mapper["attribute"]
            for edge_mapper in self.ATTRIBUTE_TO_EDGES
        ]
        elements = {
            element_id: {
                key: value
                for key, value in element_data.items()
                if key not in filtered_keys
            }
            # check that elements actually have data
            for element_id, element_data in elements.items() if len(element_data.items()) > 0
        }

        relationship_element_ids = {
            element_id
            for element_id, element in elements.items()
            if "relatedElement" in element
        }
        non_relationship_element_ids = set(elements).difference(
            relationship_element_ids
        )

        relationships = [
            elements[element_id]
            for element_id in relationship_element_ids
        ]

        graph = nx.MultiDiGraph()
        if merge:
            graph.add_nodes_from(self.graph)
            graph.add_edges_from(self.graph)

        old_graph = self.graph
        del globals()[old_graph]

        graph.add_nodes_from(
            {
                id_: elements[id_]
                for id_ in non_relationship_element_ids
            }.items()
        )
        graph.add_edges_from([
            [
                relation["relatedElement"][0]["@id"],  # source node (str id)
                relation["relatedElement"][1]["@id"],  # target node (str id)
                relation["@type"],                     # edge type (str name)
                relation,                              # edge data (dict)
            ]
            for relation in relationships
        ] + new_edges)

        with self.hold_trait_notifications():
            self.nodes = dict(graph.nodes)
            self.edges = dict(graph.edges)

            self.node_types = tuple(sorted({
                node["@type"]
                for node in self.nodes.values()
                if "@type" in node
            }))
            self.edge_types = tuple(sorted({
                edge["@type"]
                for edge in self.edges.values()
                if "@type" in edge
            }))

            self.nodes_by_type = {
                node_type: [
                    node
                    for node in graph.nodes
                    if graph.nodes[node].get("@type") == node_type
                ]
                for node_type in self.node_types
            }
            self.edges_by_type = {
                edge_type: [
                    edge
                    for edge in graph.edges
                    if edge[2] == edge_type
                ]
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

        function_attributes = getattr(self.adapt, "__annotations__", [])
        return {
            key: value
            for key, value in instructions.items()
            if key in function_attributes
        }

    def get_implied_feedforward_edges(self) -> list:
        eeg: nx.DiGraph = self.get_projection("Expression Evaluation Graph")

        edge_dict = {
            edge["@id"]: edge
            for edge in self.edges.values()
        }
        return_parameter_memberships = [
            self.edges[(source, target, kind)]
            for source, target, kind in self.edges
            if kind == "ReturnParameterMembership"
        ]

        implied_edges = []
        for membership in return_parameter_memberships:
            for result_feeder_id in eeg.predecessors(membership["memberElement"]["@id"]):
                result_feeder = self.nodes[result_feeder_id]
                rf_metatype = result_feeder["@type"]
                # we only want Expressions that have at least one input parameter
                if "Expression" not in rf_metatype or rf_metatype in ["FeatureReferenceExpression"]:
                    continue

                expr_results = []
                expr_members, para_members, result_members = [], [], []
                # assume that the members of an expression that are themselves members are
                # referenced in the same order as parameters - results of an expression
                # should feed into the input parameter owned by its owner

                owned_memberships = result_feeder["ownedMembership"]

                # NOTE: There is a special case for when there is a ResultExpressionMembership:
                # A ResultExpressionMembership is a FeatureMembership that indicates that the
                # ownedResultExpression provides the result values for the Function or Expression
                # that owns it. The owning Function or Expression must contain a BindingConnector
                # between the result parameter of the ownedResultExpression and the result
                # parameter of the Function or Expression.
                rem_flag = False
                rem_owned_ele, rem_owning_type = {}, {}
                for om_id in owned_memberships:
                    relationship = edge_dict[om_id["@id"]]
                    relationship_metatype = relationship["@type"]
                    edge_member_id = relationship["memberElement"]["@id"]
                    if "Parameter" in relationship_metatype:
                        if "ReturnParameter" in relationship_metatype:
                            result_members.append(edge_member_id)
                        else:
                            para_members.append(edge_member_id)
                    elif "Result" in relationship_metatype:
                        rem_owning_type = self.nodes[relationship["owningType"]["@id"]]
                        rem_owned_ele = self.nodes[relationship["ownedMemberElement"]["@id"]]
                        rem_flag = True
                    elif "Membership" in relationship_metatype:
                        edge_member = relationship["memberElement"]["@id"]
                        expr_members.append(edge_member)
                        if "result" in self.nodes[edge_member]:
                            expr_result = self.nodes[edge_member]["result"]["@id"]
                            expr_results.append(expr_result)

                # FIXME: streamline / simplify this
                if rem_flag and rem_owned_ele and rem_owning_type:
                    rem_cheat_expr = rem_owned_ele["@id"]
                    rem_cheat_result = rem_owned_ele["result"]["@id"]
                    rem_cheat_para = rem_owning_type["result"]["@id"]

                    expr_members = [rem_cheat_expr]
                    expr_results = [rem_cheat_result]
                    para_members = [rem_cheat_para]

                implied_edges += [
                    (expr_results[index], para_members[index], "ImpliedParameterFeedforward")
                    for index, expr in enumerate(expr_members)
                    if index < len(expr_results) and index < len(para_members)
                ]

        return implied_edges

    def get_implied_edges(self):
        new_edges = []
        for edge_generator_name in self.IMPLIED_GENERATORS:
            edge_generator = getattr(self, edge_generator_name)
            new_edges += [
                self._make_nx_multi_edge(source, target, metatype, label=metatype)
                for source, target, metatype in edge_generator()
            ]
        return new_edges

    def get_implied_elements(self):
        return {
            data["@id"]: data
            for *_, data in self.get_implied_edges()
        }

    def get_projection(self, projection: str) -> ty.Union[nx.Graph, nx.DiGraph]:
        return self.adapt(**self.get_projection_instructions(
            projection=projection,
        ))

    def adapt(
        self,
        excluded_node_types: ty.Union[list, set, tuple] = None,
        excluded_edge_types: ty.Union[list, set, tuple] = None,
        reversed_edge_types: ty.Union[list, set, tuple] = None,
    ) -> ty.Union[nx.Graph, nx.DiGraph]:
        """
            Using the existing graph, filter by node and edge types, and/or
            reverse certain edge types.
        """
        excluded_edge_types = excluded_edge_types or []
        excluded_node_types = excluded_node_types or []
        reversed_edge_types = reversed_edge_types or []

        return self._adapt(
            excluded_edge_types=tuple(sorted(excluded_edge_types)),
            excluded_node_types=tuple(sorted(excluded_node_types)),
            reversed_edge_types=tuple(sorted(reversed_edge_types)),
        ).copy()

    @lru_cache
    def _adapt(
        self,
        excluded_node_types: ty.Union[list, set, tuple] = None,
        excluded_edge_types: ty.Union[list, set, tuple] = None,
        reversed_edge_types: ty.Union[list, set, tuple] = None,
    ) -> nx.Graph:
        graph = self.graph

        mismatched_node_types = set(excluded_node_types).difference(self.node_types)
        if mismatched_node_types:
            warn(f"These node types are not in the graph: {mismatched_node_types}.")

        mismatched_edge_types = {
            *excluded_edge_types,
            *reversed_edge_types,
        }.difference(self.edge_types)
        if mismatched_edge_types:
            warn(f"These edge types are not in the graph: {mismatched_edge_types}.")

        included_nodes = sum(
            [
                nodes
                for node_type, nodes in self.nodes_by_type.items()
                if node_type not in excluded_node_types
            ],
            [],
        )
        subgraph = graph.__class__(graph.subgraph(included_nodes))

        def _process_edge(src, tgt, typ, rev_types):
            data = deepcopy(self.graph.edges.get((src, tgt, typ), {}))
            if typ in rev_types:
                tgt, src = src, tgt
                typ += "^-1"
                data["@type"] = typ
            return src, tgt, typ, data

        edges = [
            _process_edge(src, tgt, typ, reversed_edge_types)
            for src, tgt, typ in subgraph.edges
            if typ not in excluded_edge_types
        ]

        new_graph = graph.__class__()
        new_graph.add_edges_from(edges)
        new_graph.update(nodes={
            node: self.graph.nodes.get(node)
            for node in new_graph.nodes
        }.items())
        return new_graph

    @staticmethod
    def _make_undirected(graph):
        if graph.is_multigraph():
            return nx.MultiGraph(graph)
        else:
            return nx.Graph(graph)

    def get_path_graph(
        self,
        graph: nx.Graph,
        source: str,
        target: str,
        enforce_directionality: bool = True,
        try_reverse: bool = True,
    ):
        """Make a new graph with the shortest paths between two nodes"""
        new_graph = graph if enforce_directionality else self._make_undirected(graph)
        try:
            nodes = set(sum(map(
                list,
                nx.all_shortest_paths(new_graph, source, target)
            ), []))
            return graph.__class__(graph.subgraph(nodes))

        except (nx.NetworkXError, nx.NetworkXException) as exc:
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
        seed_nodes = {
            id_: max_distance
            for id_ in seeds
            if id_ in graph.nodes
        }
        seed_elements = [
            self.elements_by_id[id_]
            for id_ in seeds
            if id_ not in seed_nodes
            and id_ in self.elements_by_id
        ]
        seed_edges = [
            [element["relatedElement"][0]["@id"],
             element["relatedElement"][1]["@id"]]
            for element in seed_elements
            if len(element.get("relatedElement")) == 2
        ]

        distances = {
            node: max_distance - 1
            for node in set(sum(seed_edges, []))
            if node in graph
        }
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
