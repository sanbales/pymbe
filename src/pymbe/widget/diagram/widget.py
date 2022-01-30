import asyncio
import traceback
from warnings import warn

import ipyelk
import ipywidgets as ipyw
import networkx as nx
import traitlets as trt
from ipyelk.pipes.base import PipeDisposition

from ...graph import SysML2LabeledPropertyGraph
from ..core import BaseWidget
from .loader import SysmlLoader
from .tools import Toolbar
from .utils import Mapper


# TODO: Remove this when https://github.com/jupyrdf/ipyelk/pull/98 gets merged
class Diagram(ipyelk.Diagram):
    """A slightly modified ipyelk Diagram to avoid issue with observer"""

    @trt.observe("tools")
    def _update_tools(self, change=None):
        # TODO: Submit PR with fix to ipyelk
        if change and isinstance(change.old, tuple):
            for tool in change.old:
                tool.tee = None
                tool.on_done = None

        for tool in self.tools:
            tool.tee = self.pipe
            tool.on_done = self.refresh


@ipyw.register
class M1Viewer(ipyw.Box, BaseWidget):  # pylint: disable=too-many-ancestors
    """An ipywidget to interact with a SysML2 model through an LPG."""

    description: str = trt.Unicode("M1 Diagram").tag(sync=True)
    icon_class: str = trt.Unicode("jp-FileIcon").tag(sync=True)

    drawn_graph: nx.Graph = trt.Instance(
        nx.Graph,
        args=(),
        help="The networkx labelled property graph to be drawn.",
    )
    elk_diagram: ipyelk.Diagram = trt.Instance(ipyelk.Diagram)

    id_mapper: Mapper = trt.Instance(Mapper, args=())

    lpg: SysML2LabeledPropertyGraph = trt.Instance(
        SysML2LabeledPropertyGraph,
        args=(),
        help="The LPG of the project currently loaded.",
    )

    loader: SysmlLoader = trt.Instance(
        SysmlLoader,
        args=(),
        help="The customized ipyelk loader for transforming the SysML LPG to ELK JSON",
    )

    _task: asyncio.Future = None

    @trt.validate("children")
    def _validate_children(self, proposal: trt.Bunch):
        children = proposal.value
        if not children:
            children = [self.elk_diagram]
        return children

    def update(self, change: trt.Bunch):
        if not change.new:
            return

        self.drawn_graph = nx.Graph()
        self.lpg.model = change.new
        toolbar: Toolbar = self.elk_diagram.toolbar

        toolbar.package_selector.options = {
            pkg.name: pkg for pkg in sorted(self.lpg.model.packages)
        }

        toolbar.update_dropdown_options(
            selector="nodes",
            options={
                f"{node_type} [{len(nodes)}]": nodes
                for node_type, nodes in sorted(self.lpg.nodes_by_type.items())
            },
        )

        toolbar.update_dropdown_options(
            selector="edges",
            options={
                f"{edge_type} [{len(edges)}]": edges
                for edge_type, edges in sorted(self.lpg.edges_by_type.items())
                if edge_type in self.lpg.edge_types
            },
        )

    @trt.default("id_mapper")
    def _make_id_mapper(self) -> Mapper:
        elements = self.elk_diagram.source.index.elements
        return Mapper(
            to_sysml={
                elk_id: element.metadata.sysml_id
                for elk_id, element in elements.items()
                if hasattr(element.metadata, "sysml_id")
            },
        )

    @trt.default("elk_diagram")
    def _make_diagram(self):
        view = ipyelk.diagram.SprottyViewer()
        view.selection.observe(self._update_selected, "ids")

        tools = [
            view.selection,
            view.fit_tool,
            view.center_tool,
            ipyelk.tools.PipelineProgressBar(),
        ]

        toolbar = Toolbar(
            layout=dict(height="auto", width="auto", visibility="visible"),
            tools=tools,
            update_diagram=self._on_update_diagram_button_click,
        )
        trt.dlink(
            (self.lpg, "sysml_projections"),
            (toolbar.projection_selector, "options"),
            lambda x: tuple(x),  # pylint: disable=unnecessary-lambda
        )
        toolbar.projection_selector.options = tuple(self.lpg.sysml_projections)
        toolbar._update_children()
        toolbar.log_out = self.log_out

        # TODO: after ipyelk fix revert this back to ipyelk.Diagram
        diagram = Diagram(
            layout=dict(width="100%"),
            toolbar=toolbar,
            tools=tools,
            view=view,
        )
        trt.link((diagram, "tools"), (toolbar, "tools"))
        trt.link((diagram, "symbols"), (view, "symbols"))

        def update_mapper(change: trt.Bunch):
            if change.new == PipeDisposition.done:
                self.id_mapper = self._make_id_mapper()
                warn("Updating the id_mapper!")

        diagram.pipe.observe(update_mapper, "disposition")

        return diagram

    @trt.default("layout")
    def _default_layout(self):
        return dict(
            height="100%",
            width="auto",
        )

    @property
    def excluded_edge_types(self):
        included_edges = self.elk_diagram.toolbar.edge_type_selector.value
        if not included_edges:
            return []

        return set(self.lpg.edge_types).difference((edges[0][2] for edges in included_edges))

    @property
    def excluded_node_types(self):
        included_nodes = self.elk_diagram.toolbar.node_type_selector.value
        if not included_nodes:
            return []

        return set(self.lpg.node_types).difference(
            (self.lpg.graph.nodes[nodes[0]]["@type"] for nodes in included_nodes)
        )

    @property
    def selected_by_type_node_ids(self):
        return tuple(set(sum(map(list, self.elk_diagram.toolbar.node_type_selector.value), [])))

    @property
    def selected_by_type_nodes(self):
        return tuple(
            self.lpg.graph.nodes[id_]
            for id_ in sorted(self.selected_by_type_node_ids)
            if id_ in self.lpg.graph.nodes
        )

    @property
    def selected_by_type_edge_ids(self):
        return tuple(set(sum(map(list, self.elk_diagram.toolbar.edge_type_selector.value), [])))

    @property
    def selected_by_type_edges(self):
        return tuple(
            self.lpg.graph.edges[id_]
            for id_ in sorted(self.selected_by_type_edge_ids)
            if id_ in self.lpg.graph.edges
        )

    def _on_update_diagram_button_click(self, button: ipyw.Button):
        """Create asynchronous refresh task"""
        if self._task:
            self._task.cancel()

        def post_run(future: asyncio.Task):
            with self.log_out:
                failed = False
                try:
                    future.exception()
                except asyncio.CancelledError:
                    print("Diagram update was cancelled!")
                except Exception as exc:
                    failed = True
                    raise exc
                finally:
                    button.disabled = failed

        self._task = task = asyncio.create_task(self._update_diagram_button_click(button))
        task.add_done_callback(post_run)

    async def _update_diagram_button_click(self, button: ipyw.Button) -> asyncio.Task:
        with self.log_out:
            button.disabled = failed = True
            try:
                self._update_drawn_graph(button=button)
                failed = False
            except Exception:  # pylint: disable=broad-except
                failed = True
                warn(f"Button click for {button} failed: {traceback.format_exc()}")
            finally:
                button.disabled = failed

    def _update_drawn_graph(self, button: ipyw.Button) -> bool:
        failed = False
        toolbar: Toolbar = self.elk_diagram.toolbar
        lpg = self.lpg

        enforce_directionality = toolbar.enforce_directionality.value
        reversed_edges = toolbar.edge_type_reverser.value

        if reversed_edges and not enforce_directionality:
            raise ValueError(
                f"Reversing edge types: {sum(reversed_edges, [])} makes "
                "no sense since edges are not being enforced."
            )

        instructions = lpg.get_projection_instructions(
            projection=toolbar.projection_selector.value,
        )
        new_graph = lpg.adapt(
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
                *reversed_edges,
            },
            implied_edge_types={*instructions.get("implied_edge_types", [])},
            included_packages=toolbar.package_selector.value,
        )

        if button is toolbar.filter_to_path:
            source, target = self.selected
            new_graph = lpg.get_path_graph(
                graph=new_graph,
                source=source,
                target=target,
                enforce_directionality=enforce_directionality,
            )
            if not new_graph:
                failed = True
                warn(
                    "Could not find path between "
                    f"""{" and ".join(self.selected)}, with directionality"""
                    + (" not " if not toolbar.enforce_directionality else " ")
                    + "enforced."
                )
        elif button is toolbar.filter_by_dist:
            new_graph = lpg.get_spanning_graph(
                graph=new_graph,
                seeds=self.selected,
                max_distance=toolbar.max_distance.value,
                enforce_directionality=enforce_directionality,
            )
            if not new_graph:
                failed = True
                warn(
                    "Could not find a spanning graph of distance "
                    f"{toolbar.max_distance.value} from these seeds: "
                    f"{self.selected}."
                )

        self.drawn_graph = new_graph
        button.disabled = failed

    @trt.observe("drawn_graph")
    def _push_drawn_graph(self, change: trt.Bunch = None):
        if change is None:
            old, new = None, self.drawn_graph
        else:
            old, new = change.old, change.new

        if new == old:
            return

        def update_graph(loader, diagram, new, old):
            with self.log_out:
                diagram.style = loader.part_diagram.style
                diagram.symbols = loader.part_diagram.symbols
                diagram.source = loader.load_from_graphs(new=new, old=old)

        update_graph(
            loader=self.loader,
            diagram=self.elk_diagram,
            new=new,
            old=old,
        )

    @trt.observe("selected")
    def _update_based_on_selection(self, *_):
        """Update toolbar buttons based on selection status."""
        with self.log_out:
            selected = self.selected
            toolbar = self.elk_diagram.toolbar
            toolbar.filter_to_path.disabled = len(selected) != 2 or not all(
                isinstance(node_id, str) for node_id in selected
            )
            toolbar.filter_by_dist.disabled = not selected

    @trt.observe("selected")
    def _update_diagram_selections(self, *_):
        """Update diagram selected nodes based on app selections."""
        with self.log_out:
            view_selector = self.elk_diagram.view.selection
            diagram_elements = list(view_selector.get_index().elements.elements)

            new_selections = [
                id_ for id_ in self._map_selections(*self.selected) if id_ in diagram_elements
            ]
            if set(view_selector.ids).symmetric_difference(new_selections):
                view_selector.ids = new_selections

    def _update_selected(self, *_):
        """Updated app selections based on diagram selections."""
        with self.log_out:
            new_selections = [
                id_
                for id_ in self._map_selections(*self.elk_diagram.view.selection.ids)
                if id_ in self.model.elements
            ]
            if set(self.selected).symmetric_difference(new_selections):
                self.selected = new_selections

    def _map_selections(self, *selections: str) -> tuple:
        if not selections:
            return ()
        new_selections = self.id_mapper.get(*selections)

        # FIXME: Figure a better way to not have to do this check
        if selections and not new_selections:
            self.id_mapper = self._make_id_mapper()
            new_selections = self.id_mapper.get(*selections)
        return new_selections

    # TODO: Bring this back when the layout options are back in the toolbar
    # NOTE: Dane says the layout widget should be re-thought and support diagram hierarchy
    # @trt.observe("elk_layout")
    # def _update_observers_for_layout(self, change: trt.Bunch):
    #     if change.old not in (None, trt.Undefined):
    #         change.old.unobserve(self._element_type_opt_change)
    #         del change.old
    #     change.new.observe(self._element_type_opt_change, "value")
