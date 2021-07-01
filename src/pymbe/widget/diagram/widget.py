import typing as ty

import ipywidgets as ipyw
import networkx as nx
import traitlets as trt

import ipyelk
from ipyelk.elements import layout_options as opt

from ...graph import SysML2LabeledPropertyGraph
from ..core import BaseWidget
from .parts import Part, PartDiagram
from .relationships import METATYPE_TO_RELATIONSHIP_TYPES, DirectedAssociation
from .tools import Toolbar
from .utils import Mapper

opt.OptionsWidget

@ipyw.register
class SysML2LPGWidget(SysML2LabeledPropertyGraph, ipyelk.Diagram, BaseWidget):
    """An ipywidget to interact with a SysML2 model through an LPG."""

    description = trt.Unicode("Diagram").tag(sync=True)

    diagram: PartDiagram = trt.Instance(PartDiagram, args=())
    diagram_graph: nx.Graph = trt.Instance(nx.Graph, args=())

    id_mapper: Mapper = trt.Instance(Mapper, args=())
    parts: ty.Dict[str, Part] = trt.Dict(
        key_trait=trt.Unicode(),
        value_trait=trt.Instance(Part),
    )

    loader: ipyelk.ElementLoader = trt.Instance(
        ipyelk.ElementLoader,
        kw=dict(
            default_node_opts={
                opt.Direction.identifier: opt.Direction(value="RIGHT").value,
                opt.HierarchyHandling.identifier: opt.HierarchyHandling().value,
            },
        ),
    )

    toolbar: Toolbar = trt.Instance(Toolbar, args=())

    @trt.default("id_mapper")
    def _make_id_mapper(self) -> Mapper:
        return Mapper(
            to_map={
                elk_id: getattr(element, "data", {}).get("@id")
                for elk_id, element in self.source.index.elements.items()
            },
        )

    @trt.validate("toolbar")
    def _validate_proj_selector(self, proposal: trt.Bunch) -> ipyw.Dropdown:
        toolbar = proposal.value

        # TODO: Do other things to the toolbar

        projection_selector = toolbar.projection_selector
        if not projection_selector.options:
            projection_selector.options = tuple(self.sysml_projections)
        projection_selector.rows = self.max_type_selector_rows
        return toolbar

    @trt.validate("layout")
    def _validate_layout(self, proposal):
        layout = proposal.value
        layout.height = "100%"
        layout.width = "auto"
        return layout

    @trt.observe("source")
    def _update_mapper(self, *_):
        self.id_mapper = self._make_id_mapper()

    @trt.observe("sysml_projections")
    def _update_projection_selector(self, *_):
        self.projection_selector.options = tuple(self.sysml_projections)

    @trt.observe("nodes_by_type", "edges_by_type")
    def _update_options_for_toolbar_selectors(self, change: trt.Bunch):
        if not change or not change.new:
            return
        selector, *_ = change.name.split("_")
        self.toolbar.update_options(
            selector=selector,
            options={
                f"{item_type} [{len(items)}]": items
                for item_type, items in sorted(change.new.items())
            },
        )

    @trt.observe("selected")
    def _update_based_on_selection(self, *_):
        selected = self.selected
        self.filter_to_path.disabled = (
            len(selected) != 2 or
            not all(isinstance(n, str) for n in selected)
        )
        self.filter_by_dist.disabled = not selected

    @property
    def excluded_edge_types(self):
        included_edges = self.toolbar.edge_type_selector.value
        if not included_edges:
            return []

        return set(self.edge_types).difference((
                edges[0][2] for edges in included_edges
            ))

    @property
    def excluded_node_types(self):
        included_nodes = self.toolbar.node_type_selector.value
        if not included_nodes:
            return []

        return set(self.node_types).difference((
            self.graph.nodes[nodes[0]]["@type"] for nodes in included_nodes
        ))

    @property
    def selected_by_type_node_ids(self):
        return tuple(set(sum(
            map(
                list,
                self.toolbar.node_type_selector.value
            ),
            []
        )))

    @property
    def selected_by_type_nodes(self):
        return tuple(
            self.graph.nodes[id_]
            for id_ in sorted(self.selected_by_type_node_ids)
            if id_ in self.graph.nodes
        )

    @property
    def selected_by_type_edge_ids(self):
        return tuple(set(sum(
            map(
                list,
                self.toolbar.edge_type_selector.value
            ),
            []
        )))

    @property
    def selected_by_type_edges(self):
        return tuple(
            self.graph.edges[id_]
            for id_ in sorted(self.selected_by_type_edge_ids)
            if id_ in self.graph.edges
        )

    def _refresh_button_click(self, button: ipyw.Button):
        button.disabled = failed = True
        try:
            failed = self._update_diagram_graph(button=button)
        except Exception as exc:
            self.log.warning(f"Button click for {button} failed: {exc}")
        finally:
            button.disabled = failed

    def _update_diagram_graph(self, button=None):
        failed = False

        enforce_directionality = self.enforce_directionality.value
        if self.edge_type_reverser.value and not enforce_directionality:
            raise ValueError(
                f"Reversing edge types: {self.edge_type_reverser.value} "
                "makes no sense since edges are not being enforced."
            )

        instructions: dict = self.get_projection_instructions(
            projection=self.projection_selector.value,
        )
        new_graph = self.adapt(
            excluded_edge_types={
                *instructions.get("excluded_edge_types", []),
                *self.excluded_edge_types,
            },
            excluded_node_types={
                *instructions.get("excluded_node_types", []),
                *self.excluded_node_types,
            },
            reversed_edge_types={
                *instructions.get("reversed_edge_types", []),
                *self.edge_type_reverser.value,
            },
        )

        if button is self.filter_to_path:
            source, target = self.selected
            new_graph = self.get_path_graph(
                graph=new_graph,
                source=source,
                target=target,
                enforce_directionality=enforce_directionality,
            )
            if not new_graph:
                failed = True
                self.log.warning(
                    "Could not find path between " 
                    f"""{" and ".join(self.selected)}, with directionality """
                    f"""{"not" if not self.enforce_directionality else ""} """ 
                    "enforced."
                )
        elif button is self.filter_by_dist:
            new_graph = self.get_spanning_graph(
                graph=new_graph,
                seeds=self.selected,
                max_distance=self.max_distance.value,
                enforce_directionality=enforce_directionality,
            )
            if not new_graph:
                failed = True
                self.log.warning(
                    "Could not find a spanning graph of distance "
                    f"{self.max_distance.value} from these seeds: " 
                    f"{self.selected}."
                )

        self._update_diagram(new_graph=new_graph)
        # self.diagram.elk_app.refresh()
        # self.diagram.elk_app.diagram.fit()
        return failed

    @trt.observe("diagram")
    def _update_diagram(self, *_):
        diagram = self.diagram
        self.source = self.loader.load(diagram)
        self.style = diagram.style
        self.view.symbols = diagram.symbols

    def _update_diagram(self, new_graph: nx.Graph):
        diagram = PartDiagram()
        parts = self.parts
        new_parts = {
            id_: Part.from_data(node_data)
            for id_, node_data in new_graph.nodes.items()
            if node_data
            and id_ not in parts
        }
        parts.update(new_parts)

        for (source, target, metatype), edge_data in new_graph.edges.items():
            if source not in parts:
                self.log.warn(
                    f"Could not map source: {source} in "
                    f"'{metatype}' with {target}"
                )
                continue

            if target not in parts:
                self.log.warn(
                    f"Could not map target: {target} in "
                    f"'{metatype}' with {source}"
                )
                continue

            _ = diagram.add_relationship(
                source=parts[source],
                target=parts[target],
                cls=METATYPE_TO_RELATIONSHIP_TYPES.get(
                    metatype,
                    DirectedAssociation,
                ),
                data=edge_data,
            )

        self.diagram = diagram

    @trt.observe("selected")
    def _update_diagram_selections(self, *_):
        new_selections = self._map_selections(*self.selected)
        view_selector = self.view.selection
        if set(view_selector.ids).symmetric_difference(new_selections):
            view_selector.ids = new_selections

    def _update_selected(self, *_):
        new_selections = self._map_selections(*self.view.selection.ids)
        if set(self.selected).symmetric_difference(new_selections):
            self.selected = new_selections

    def _map_selections(self, *selections: str) -> tuple:
        if not selections:
            return ()
        new_selections = self.id_mapper.get(*selections)
        if selections and not new_selections:
            self.id_mapper = self._make_id_mapper()
            new_selections = self.id_mapper.get(*selections)
        return tuple(new_selections)

    @trt.observe("elk_layout")
    def _update_observers_for_layout(self, change: trt.Bunch):
        if change.old not in (None, trt.Undefined):
            change.old.unobserve(self._element_type_opt_change)
            del change.old
        change.new.observe(self._element_type_opt_change, "value")
