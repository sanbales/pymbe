import ipytree as ipyt
import ipywidgets as ipyw
import traitlets as trt
import typing as ty

from .core import BaseWidget


class Element(ipyt.Node):
    """A project element node compatible with ipytree."""

    _identifier: str = trt.Unicode()
    _owner: str = trt.Unicode("self", allow_none=True)
    _type: str = trt.Unicode()
    _data: dict = trt.Dict()


@ipyw.register
class ContainmentTree(ipyt.Tree, BaseWidget):
    """A widget to explore the structure and data in a project."""

    description: str = trt.Unicode("Containment Tree").tag(sync=True)

    default_icon: str = trt.Unicode("genderless").tag(sync=True)

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
            MultiplicityRange="ellipsis-h",  # "star-of-life" or "share-alt-square"
            OperatorExpression="broom",  # "hashtag"
            Package="archive",  # "cube" or "box" or "box-open"
            PartDefinition="file-powerpoint",
            PartUsage="copy",
            ReferenceUsage="code-branch",
            StateDefinition="file-contract",
            Succession="long-arrow-alt-right",
        ),
    ).tag(sync=True)

    nodes_by_id: ty.Dict[str, Element] = trt.Dict(
        key_trait=trt.Unicode(),
        value_trait=trt.Instance(Element),
        kw={},
    )

    @trt.validate("layout")
    def _validate_layout(self, proposal: trt.Bunch) -> ipyw.Layout:
        layout = proposal.value
        layout.overflow_y = "auto"
        layout.width = "auto"
        return layout

    @trt.observe("selected_nodes")
    def _update_selected(self, *_):
        element_ids = {node._identifier for node in self.selected_nodes}
        self.update_selected(*element_ids)

    @trt.observe("selected")
    def _update_selected_nodes(self, *_):
        nodes_selected = {
            node._identifier
            for node in self.selected_nodes
        }
        if not nodes_selected.symmetric_difference(self.selected):
            return
        with self.hold_trait_notifications():
            if not self.selected:
                self.deselect_nodes()
                return

            nodes_to_deselect = [
                node
                for node in self.selected_nodes
                if node._identifier not in self.selected
            ]
            if nodes_to_deselect:
                self.deselect_nodes(*nodes_to_deselect)

            self.select_nodes(*self.selected)

    @trt.observe("nodes_by_id")
    def _update_tree(self, *_):
        # find the root nodes and sort them
        roots = self.sort_nodes([
            node for node in self.nodes_by_id.values()
            if node._owner is None
        ])

        # update the tree
        self._clear_tree()
        for root in roots:
            self.add_node(root)

    @trt.observe("icons_by_type", "default_icon")
    def _update_icons(self, *_):
        for id_, element_data in self.elements_by_id.items():
            node = self.nodes_by_id[id_]
            node.icon = self.icons_by_type.get(
                element_data["@type"],
                self.default_icon,
            )

    def update(self, elements: dict = None):
        elements = elements or self.elements_by_id
        nodes = {
            element_id: self._make_node(element=element)
            for element_id, element in elements.items()
        }
        for node in nodes.values():
            if node._owner in nodes:
                nodes[node._owner].add_node(node)

        # Filter nodes to those that have subnodes or a proper name
        nodes = {
            id_: node
            for id_, node in nodes.items()
            if node.nodes
            or node._owner
            or node._data.get("name", None)
        }
        # Sort the child nodes
        for node in nodes.values():
            node.nodes = self.sort_nodes(node.nodes)

        with self.hold_trait_notifications():
            self.update_selected()
            self.nodes_by_id = nodes

    def select_nodes(self, *nodes: str):
        """Select a list of nodes"""
        for node_id in nodes:
            node = self.nodes_by_id.get(node_id, None)
            if node:
                node.selected = True

    def deselect_nodes(self, *nodes: str):
        """Deselect a node, or deselect all if none is specified."""
        if not nodes:
            nodes = [
                node
                for node_id, node in self.nodes_by_id.items()
                if node.selected
            ]
        for node in nodes:
            node.selected = False

    def _clear_tree(self):
        for node in self.nodes:
            self.remove_node(node)
            del node

    def _make_node(self, element):
        element_id = element["@id"]

        owner = (
            element.get("owner", None)
            or element.get("owningRelatedElement", None)
            or {}
        ).get("@id", None)

        return Element(
            icon=self.icons_by_type.get(element["@type"], self.default_icon),
            name=(
                element["name"]
                or f"""«{element["@type"]}: {element_id}»"""
            ),
            _data=element,
            _identifier=element_id,
            _owner=owner,
            _type=element["@type"],
        )

    @staticmethod
    def sort_nodes(nodes: (list, tuple, set)) -> tuple:
        # Sort nodes with number of subnodes first and then by name
        return tuple(sorted(
            nodes,
            key=lambda n: (-len(n.nodes), n.name),
        ))
