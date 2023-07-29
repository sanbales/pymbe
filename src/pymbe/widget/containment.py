import asyncio
import typing as ty
from pathlib import Path

import ipytree as ipyt
import ipywidgets as ipyw
import traitlets as trt
from ipylab import JupyterFrontEnd

from ..model import Element, Model
from .client import APIClientWidget, FileLoader
from .core import BaseWidget

__all__ = ("ContainmentTree",)


class ElementNode(ipyt.Node):
    """A project element node compatible with ipytree."""

    _data: dict = trt.Dict()
    _element: Element = trt.Instance(Element, allow_none=True)
    _identifier: str = trt.Unicode()
    _metatype: str = trt.Unicode()
    _owner: str = trt.Unicode(allow_none=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.close_icon = "caret-down"
        self.open_icon = "caret-right"


@ipyw.register
class ContainmentTree(ipyw.VBox, BaseWidget):
    """A widget to explore the structure and data in a project."""

    app: JupyterFrontEnd = trt.Instance(JupyterFrontEnd, allow_none=True)

    description: str = trt.Unicode("Containment Tree").tag(sync=True)
    icon_class: str = trt.Unicode("jp-TreeViewIcon").tag(sync=True)

    api_client: APIClientWidget = trt.Instance(APIClientWidget)
    file_loader: FileLoader = trt.Instance(FileLoader, args=())

    default_icon: str = trt.Unicode("genderless").tag(sync=True)
    indeterminate_icon: str = trt.Unicode("question").tag(sync=True)

    tree = trt.Instance(ipyt.Tree, kw=dict(layout=dict(overflow_y="auto")))
    add_widget: ty.Callable = trt.Callable(allow_none=True)

    launch_file_loader: ipyw.Button = trt.Instance(
        ipyw.Button,
        kw=dict(
            icon="folder-open",
            layout=dict(width="40px"),
            tooltip="Launch file loader for SysML v2 models",
        ),
    )
    launch_api: ipyw.Button = trt.Instance(
        ipyw.Button,
        kw=dict(
            icon="cloud-download-alt",
            layout=dict(width="40px"),
            tooltip="Launch client for Pilot Implementation of the SysMLv2 ReST API",
        ),
    )
    pop_log: ipyw.Button = trt.Instance(
        ipyw.Button,
        kw=dict(
            icon="book",
            layout=dict(width="40px"),
            tooltip="Show log",
        ),
    )
    save_model: ipyw.Button = trt.Instance(
        ipyw.Button,
        kw=dict(
            icon="save",
            layout=dict(width="40px"),
            tooltip="Save to file",
        ),
    )

    icons_by_type: dict = trt.Dict(
        key_trait=trt.Unicode(),
        value_trait=trt.Unicode(),
        kw=dict(
            ActionUsage="copy",
            AttributeUsage="copy",  # "underline"
            Expression="code",
            Feature="terminal",  # "pencil-alt"
            Function="calculator",  # "square-root-alt"
            InvocationExpression="comment-alt",
            ItemDefinition="file-invoice",  # info
            LiteralInteger="quote-right",
            Model="map",
            MultiplicityRange="ellipsis-h",  # "star-of-life" or "share-alt-square"
            OperatorExpression="broom",  # "hashtag"
            Package="archive",  # "cube" or "box" or "box-open"
            PartDefinition="file-powerpoint",
            PartUsage="copy",
            ReferenceUsage="code-branch",
            Relationship="link",
            StateDefinition="file-contract",
            Succession="long-arrow-alt-right",
        ),
    ).tag(sync=True)

    nodes_by_id: ty.Dict[str, ElementNode] = trt.Dict(
        key_trait=trt.Unicode(),
        value_trait=trt.Instance(ElementNode),
        kw={},
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tree.observe(self._update_selected, "selected_nodes")

        self.launch_api.on_click(self._pop_api_client)
        self.launch_file_loader.on_click(self._pop_file_loader)
        self.pop_log.on_click(self._pop_log_out)
        self.save_model.on_click(self._save_to_disk)

        for linked_attribute in ("model", "log_out"):
            for widget in (self.api_client, self.file_loader):
                trt.link((self, linked_attribute), (widget, linked_attribute))

    @trt.default("api_client")
    def _make_api_client(self) -> APIClientWidget:
        api_client = APIClientWidget(host_url="http://sysml2.intercax.com")
        api_client._set_layout()
        return api_client

    @trt.default("add_widget")
    def _make_add_widget(self) -> ty.Callable:
        def add_widget(widget: ipyw.DOMWidget, mode="split-right"):
            self.app.shell.add(widget, "main", {"mode": "split-top"})

        return add_widget

    @property
    def selected_nodes(self):
        return self.tree.selected_nodes

    def _pop_file_loader(self, *_):
        with self.log_out:
            self.add_widget(self.file_loader, mode="split-top")

    def _save_to_disk(self, *_):
        with self.log_out:
            if not self.model:
                print("No model loaded!")
                return
            self.model.save_to_file(filepath=Path(".") / self.model.name)

    def _pop_api_client(self, *_):
        with self.log_out:
            self.add_widget(self.api_client, mode="split-top")

    def _pop_log_out(self, *_):
        with self.log_out:
            # TextEditorIcon or PaletteIcon
            self.log_out.add_traits(
                description=trt.Unicode("Log").tag(sync=True),
                icon_class=trt.Unicode("jp-ListIcon").tag(sync=True),
            )
            self.add_widget(self.log_out, mode="split-bottom")

    @trt.validate("children")
    def _validated_children(self, proposal: trt.Bunch) -> tuple:
        children = proposal.value
        if children:
            return children
        return tuple(
            [
                ipyw.HBox(
                    children=[
                        self.launch_file_loader,
                        self.launch_api,
                        self.save_model,
                        self.pop_log,
                    ],
                    layout=dict(min_height="50px"),
                ),
                self.tree,
            ]
        )

    @trt.validate("layout")
    def _validate_layout(self, proposal: trt.Bunch) -> ipyw.Layout:
        layout = proposal.value
        layout.overflow_y = "auto"
        layout.width = "auto"
        return layout

    def _update_selected(self, *_):
        with self.log_out:
            element_ids = {node._identifier for node in self.selected_nodes}
            self.update_selected(*element_ids)

    def _add_selected_node_lineage(self, missing_element_id: str):
        element = self.model.elements[missing_element_id]
        nodes_by_id = self.nodes_by_id
        lineage = [element]
        while element.owner and element.owner._id not in self.nodes_by_id:
            element = element.owner
            lineage += [element]

        if not element.owner:
            return

        for element in reversed(lineage):
            nodes_by_id[element._id] = node = self._make_node(element=element, opened=True)
            parent = nodes_by_id[element.owner._id]
            parent.add_node(node)
            parent.opened = True

    @trt.observe("selected")
    def _update_selected_nodes(self, *_):
        with self.log_out:
            nodes_selected = {node._identifier for node in self.selected_nodes}
            if not nodes_selected.symmetric_difference(self.selected):
                return
            with self.hold_trait_notifications():
                if not self.selected:
                    self.deselect_nodes()
                    return

                for element_id in self.selected:
                    if element_id not in self.nodes_by_id:
                        self._add_selected_node_lineage(element_id)

                nodes_to_deselect = [
                    node for node in self.selected_nodes if node._identifier not in self.selected
                ]
                if nodes_to_deselect:
                    self.deselect_nodes(*nodes_to_deselect)

                self.select_nodes(*self.selected)

    def _update_tree(self):
        # find the root nodes and sort them
        roots = self.sort_nodes(
            [node for node in self.nodes_by_id.values() if node._owner is None]
        )

        # update the tree
        self._clear_tree()
        for root in roots:
            self.tree.add_node(root)

    @trt.observe("icons_by_type", "default_icon")
    def _update_icons(self, *_):
        for node in self.nodes_by_id.values():
            node.icon = self.icons_by_type.get(
                node._element._metatype,
                self.default_icon,
            )

    def update(self, change: trt.Bunch):
        with self.log_out:
            model = change.new
            if not isinstance(model, Model):
                return

            model_id = str(model.source) or "SYSML_MODEL"
            model_name = model.name
            if model.source:
                model_name += f" ({model.source})"
            model_node = ElementNode(
                icon=self.icons_by_type["Model"],
                name=model_name,
                _data=dict(source=model.source),
                _identifier=model_id,
                _owner=None,
                _metatype="MODEL",
            )

            nodes = {
                model_id: model_node,
            }

            elements = model.ownedElement
            nodes.update(
                {
                    element._id: self._make_node(element=element, root=model_id)
                    for element in elements
                    if element._id not in nodes
                }
            )
            for node in nodes.values():
                if node._owner in nodes:
                    nodes[node._owner].add_node(node)

            # Sort the child nodes
            for node in nodes.values():
                node.nodes = self.sort_nodes(node.nodes)

            with self.hold_trait_notifications():
                self.update_selected()
                self.nodes_by_id = nodes
                self._update_tree()

    @staticmethod
    async def add_node(parent: ipyt.Node, child: ipyt.Node):
        if child not in parent.nodes:
            parent.add_node(child)
            parent.opened = True

    def _observe_node_selection(self, change: trt.Bunch = None):
        with self.log_out:
            parent_node: ElementNode = change.owner
            selected: bool = change.new

            if not selected:
                return

            if parent_node.icon == self.indeterminate_icon:
                parent_node.icon = self.icons_by_type.get(parent_node._metatype, self.default_icon)
            else:
                parent_node.unobserve(self._observe_node_selection)

            nodes = {
                element._id: self._make_node(element)
                for element in parent_node._element.get("ownedElement", [])
            }
            if nodes:
                self.nodes_by_id.update(nodes)
                for node in nodes.values():
                    asyncio.create_task(self.add_node(parent_node, node))

            parent_node.unobserve(self._observe_node_selection)

    def select_nodes(self, *nodes: str):
        """Select a list of nodes"""
        for node_id in nodes:
            node = self.nodes_by_id.get(node_id, None)
            if node:
                node.selected = True

    def deselect_nodes(self, *nodes: str):
        """Deselect a node, or deselect all if none is specified."""
        if not nodes:
            nodes = [node for node in self.nodes_by_id.values() if node.selected]
        for node in nodes:
            node.selected = False

    def _clear_tree(self):
        tree = self.tree
        for node in tree.nodes:
            tree.remove_node(node)
            del node

    def _make_node(self, element: Element, root=None, opened=False):
        node = self.nodes_by_id.get(element._id)
        if node:
            return node
        children = element.get("ownedElement", [])
        data = element._data
        metatype = element._metatype
        owner = element.get_owner()
        owner_id = getattr(owner, "_id", root)
        icon = (
            self.indeterminate_icon
            if children
            else self.icons_by_type.get(metatype, self.default_icon)
        )

        node = ElementNode(
            icon=icon,
            name=element.get("effectiveName") or f"{element._id} «{element._metatype}»",
            opened=opened,
            selected=element._id in self.selected,
            _data=data,
            _element=element,
            _identifier=data["@id"],
            _owner=owner_id,
            _metatype=metatype,
        )
        if children:
            node.observe(self._observe_node_selection, "selected")
        return node

    @staticmethod
    def sort_nodes(nodes: ty.Union[ty.List, ty.Tuple, ty.Set]) -> tuple:
        # Sort nodes with number of subnodes first and then by name
        return tuple(
            sorted(
                nodes,
                key=lambda n: (-len((n._element or {}).get("ownedElement", [])), n.name),
            )
        )
