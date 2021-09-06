import threading
import typing as ty

import ipytree as ipyt
import ipywidgets as ipyw
import traitlets as trt
from wxyz.lab import DockPop

from ..model import Element, Model
from .client import SysML2ClientWidget
from .core import BaseWidget


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

    description: str = trt.Unicode("Containment Tree").tag(sync=True)

    client: SysML2ClientWidget = trt.Instance(SysML2ClientWidget)

    default_icon: str = trt.Unicode("genderless").tag(sync=True)
    indeterminate_icon: str = trt.Unicode("question").tag(sync=True)

    tree = trt.Instance(ipyt.Tree, kw=dict(layout=dict(overflow_y="auto")))
    add_widget: ty.Callable = trt.Callable(allow_none=True)

    load_from_file: ipyw.Button = trt.Instance(
        ipyw.Button,
        kw=dict(
            icon="folder-open",
            layout=dict(width="40px"),
            tooltip="Load from file",
        ),
    )
    launch_api: ipyw.Button = trt.Instance(
        ipyw.Button,
        kw=dict(
            icon="cloud-download-alt",
            layout=dict(width="40px"),
            tooltip="Launch API",
        ),
    )
    pop_log: ipyw.Button = trt.Instance(
        ipyw.Button,
        kw=dict(
            icon="book",
            layout=dict(width="40px"),
            tooltip="Pop Log",
        ),
    )

    icons_by_type: dict = trt.Dict(
        key_trait=trt.Unicode(),
        value_trait=trt.Unicode(),
        kw=dict(
            ActionUsage="copy",
            AttributeUsage="copy",  # "underline"
            Expression="code",
            Feature="terminal",  # "pencil-alt",
            Function="square-root-alt",
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
        self.pop_log.on_click(self._pop_log_out)
        for linked_attribute in ("model", "log_out"):
            trt.link((self, linked_attribute), (self.client, linked_attribute))

    @trt.default("client")
    def _make_client(self) -> SysML2ClientWidget:
        client = SysML2ClientWidget(host_url="http://sysml2.intercax.com")
        client._set_layout()
        return client

    @property
    def selected_nodes(self):
        return self.tree.selected_nodes

    def _pop_api_client(self, *_):
        with self.log_out:
            mode = "split-top"
            if self.add_widget:
                self.add_widget(self.client, mode=mode)
            else:
                DockPop([self.client], mode=mode)

    def _pop_log_out(self, *_):
        with self.log_out:
            DockPop([self.log_out], mode="split-right")

    @trt.validate("children")
    def _validated_children(self, proposal: trt.Bunch) -> tuple:
        children = proposal.value
        if children:
            return children
        return tuple(
            [
                ipyw.HBox(
                    children=[self.load_from_file, self.launch_api],
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

    # @trt.observe("icons_by_type", "default_icon")
    # def _update_icons(self, *_):
    #     for id_, node in self.nodes_by_id.items():
    #         new_icon = node.element
    #         node.icon = self.icons_by_type.get(
    #             element_data["@type"],
    #             self.default_icon,
    #         )

    def update(self, change: trt.Bunch):
        with self.log_out:
            model = change.new
            if not isinstance(model, Model):
                return

            model_id = str(model.source) or "SYSML_MODEL"
            model_node = ElementNode(
                icon=self.icons_by_type["Model"],
                name=model.name,
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

    def _observe_node_selection(self, change: trt.Bunch = None):
        def add_node(parent, child):
            if child not in parent.nodes:
                parent.add_node(child)

        with self.log_out:
            parent_node: ElementNode = change.owner
            parent_element = parent_node._element
            elements = parent_element.get("ownedElement", [])
            if parent_node.icon == self.indeterminate_icon:
                parent_node.icon = self.icons_by_type.get(parent_node._metatype, self.default_icon)
            if not elements:
                parent_node.unobserve(self._observe_node_selection)
                return
            nodes = {element._id: self._make_node(element) for element in elements}
            self.nodes_by_id.update(nodes)
            threads = [
                threading.Thread(target=add_node, args=(parent_node, node))
                for node in nodes.values()
            ]
            # start executing the threads for adding the nodes
            _ = [thread.start() for thread in threads]

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

    def _make_node(self, element: Element, root=None):
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
            opened=False,
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
